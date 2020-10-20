from advertorch.attacks import Attack, LabelMixin
from advertorch.utils import to_one_hot
import torch
import torch.optim as optim

from torch.optim.lr_scheduler import StepLR

R = 100
L = 50000
I = int(L*0.8)

LARGE_NUM = 10000
SAMPLES_PER_ITER = 100
ADAM_LR = 0.05
SGD_LR = 10000.0


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()


def grad_reverse(x):
    return GradReverse.apply(x)


class OverPoweredAttack(Attack, LabelMixin):
    def __init__(self, predict, num_classes, gan,
                 robustness_norm, l2_square_threshold,
                 targeted=False, max_iterations=50000):
        super(OverPoweredAttack, self).__init__(predict, loss_fn=None, clip_min=0., clip_max=1.)

        self.num_classes = num_classes
        self.gan = gan
        self.robustness_norm = robustness_norm
        self.l2_square_threshold = l2_square_threshold
        self.targeted = targeted
        self.max_iterations = max_iterations

    def _look_ahead(self, gen, y_onehot):

        adv_loss_flatten = 0
        self.noise_buffer = []

        with torch.no_grad():
            for _ in range(SAMPLES_PER_ITER):
                # sample random noise of shape (R*BATCH_SIZE, C, W, H)
                r = torch.randn_like(gen)
                # renormalize the noise to unit norm
                norm_r = torch.sqrt(r.view(r.size(0), -1).pow(2).sum(-1)).view(-1, 1, 1, 1)

                noise = self.robustness_norm * r / norm_r
                output = self.predict(gen + noise)
                self.noise_buffer.append(noise)

                real = (y_onehot * output).sum(dim=1)
                other = ((1.0 - y_onehot) * output - (y_onehot * LARGE_NUM)).max(dim=1)[0]
                adv_loss_flatten += real - other

            adv_loss_flatten = adv_loss_flatten / SAMPLES_PER_ITER

        return adv_loss_flatten

    def perturb_(self, x, zh=None, y=None):
        offsets = torch.arange(0, x.size(0)).cuda() * R

        zhat = torch.repeat_interleave(zh, R, dim=0).detach()
        zhat.requires_grad_()
        x_tiled = torch.repeat_interleave(x, R, dim=0)

        y_onehot = to_one_hot(y, self.num_classes).float()
        y_onehot = torch.repeat_interleave(y_onehot, R, dim=0)

        # create a mask which checks whether attacks are done/not done
        not_dones_mask = torch.ones(zhat.shape[0])
        # initialize the dual variable/lagrange multiplier for the perturbation constraint
        LAM = 1000 * torch.ones_like(not_dones_mask, device=x.device)
        LAM.requires_grad_()

        opt = optim.Adam([zhat], lr=ADAM_LR)
        lam_opt = optim.SGD([LAM], lr=SGD_LR)
        lr_maker = StepLR(opt, step_size=I)

        LAM = grad_reverse(LAM)

        for i in range(self.max_iterations):

            gen = self.gan(zhat)
            adv_loss_flatten = self._look_ahead(gen, y_onehot)
            adv_loss = adv_loss_flatten.view(-1, R)

            l2_loss_flatten = (gen - x_tiled).pow(2).mean(dim=(1, 2, 3))
            # if the perturbation is below THR/2, don't include it in the loss, set it to some constant
            l2_loss_flatten = l2_loss_flatten * (l2_loss_flatten > self.l2_square_threshold / 2).float() - (
                        l2_loss_flatten <= self.l2_square_threshold / 2).float()
            l2_loss = l2_loss_flatten.view(-1, R)

            not_dones_mask = 1 - (l2_loss <= self.l2_square_threshold).float() * (adv_loss <= -1).float()

            # weird here. For each image, not_dones will be all 1 or all 0
            not_dones_mask = not_dones_mask.min(dim=1)[0].repeat(1, R)
            not_dones_mask = not_dones_mask.view(-1, 1)

            ind = (adv_loss + LARGE_NUM * (l2_loss > self.l2_square_threshold).float()).argmin(dim=1)
            ind = ind + offsets

            best_adv_loss = adv_loss_flatten[ind]
            best_l2_loss = l2_loss_flatten[ind]
            # evaluate and terminate early to prevent dividing by zero
            if not_dones_mask.mean() < 0.1 or i == self.max_iterations - 1:
                return gen[ind].clone().detach(), zhat[ind].clone().detach()

            print("----")
            print("Norms", best_l2_loss.item())
            print("Losses", best_adv_loss.item())
            print("Success rate: ", 1. - not_dones_mask.mean())
            print("Lambda: ", LAM)

            not_dones_mask = not_dones_mask / not_dones_mask.mean()

            opt.zero_grad()
            lam_opt.zero_grad()

            for noise in self.noise_buffer:
                gen = self.gan(zhat)
                output = self.predict(gen + noise)
                real = (y_onehot * output).sum(dim=1)
                other = ((1.0 - y_onehot) * output - y_onehot * LARGE_NUM).max(dim=1)[0]

                loss = (real - other) / SAMPLES_PER_ITER
                ((loss * not_dones_mask).mean()).backward()

            gen = self.gan(zhat)
            l2_loss_flatten = (gen - x_tiled).pow(2).mean(dim=(1, 2, 3))
            # if the perturbation is below THR/2, don't include it in the loss, set it to some constant
            l2_loss_flatten = l2_loss_flatten * (l2_loss_flatten > self.l2_square_threshold / 2).float() - (
                    l2_loss_flatten <= self.l2_square_threshold / 2).float()
            ((LAM * l2_loss_flatten * not_dones_mask).mean()).backward()
            opt.step()
            lam_opt.step()
            lr_maker.step()

    def perturb(self, x, zh=None, y=None):
        offsets = torch.arange(0, x.size(0)).cuda() * R

        zhat = torch.repeat_interleave(zh, R, dim=0).detach()
        zhat.requires_grad_()
        x_tiled = torch.repeat_interleave(x, R, dim=0)

        y_onehot = to_one_hot(y, self.num_classes).float()
        y_onehot = torch.repeat_interleave(y_onehot, R, dim=0)

        # create a mask which checks whether attacks are done/not done
        not_dones_mask = torch.ones(zhat.shape[0])
        # initialize the dual variable/lagrange multiplier for the perturbation constraint
        LAM = 1000.0 * torch.ones_like(not_dones_mask, device=x.device)
        LAM.requires_grad_()

        opt = optim.Adam([zhat], lr=ADAM_LR)
        lam_opt = optim.SGD([LAM], lr=SGD_LR)
        lr_maker = StepLR(opt, step_size=I)

        LAM = grad_reverse(LAM)

        for i in range(self.max_iterations):

            opt.zero_grad()
            lam_opt.zero_grad()

            gen = self.gan(zhat)

            l2_loss_flatten = (gen - x_tiled).pow(2).mean(dim=(1, 2, 3))
            # if the perturbation is below THR/2, don't include it in the loss, set it to some constant
            l2_loss_flatten = l2_loss_flatten * (l2_loss_flatten > self.l2_square_threshold / 2).float() - (
                        l2_loss_flatten <= self.l2_square_threshold / 2).float()
            l2_loss = l2_loss_flatten.view(-1, R)

            adv_loss_flatten = 0
            for _ in range(SAMPLES_PER_ITER):
                # sample random noise of shape (R*BATCH_SIZE, C, W, H)
                r = torch.randn_like(gen)
                # renormalize the noise to unit norm
                norm_r = torch.sqrt(r.view(r.size(0), -1).pow(2).sum(-1)).view(-1, 1, 1, 1)
                noise = self.robustness_norm * r / norm_r
                output = self.predict(gen + noise)

                real = (y_onehot * output).sum(dim=1)
                other = ((1.0 - y_onehot) * output - y_onehot * LARGE_NUM).max(dim=1)[0]

                adv_loss_flatten += (real - other)

            adv_loss_flatten = adv_loss_flatten / SAMPLES_PER_ITER

            total_loss = adv_loss_flatten.mean() + l2_loss_flatten * LAM

            adv_loss = adv_loss_flatten.view(-1, R)
            not_dones_mask = 1 - (l2_loss <= self.l2_square_threshold).float() * (adv_loss <= -1).float()
            # weird here. For each image, not_dones will be all 1 or all 0
            not_dones_mask = not_dones_mask.min(dim=1)[0].repeat(1, R)
            not_dones_mask = not_dones_mask.view(-1, 1)

            ind = (adv_loss + LARGE_NUM * (l2_loss > self.l2_square_threshold).float()).argmin(dim=1)
            ind = ind + offsets

            best_adv_loss = adv_loss_flatten[ind]
            best_l2_loss = l2_loss_flatten[ind]
            # evaluate and terminate early to prevent dividing by zero
            if not_dones_mask.mean() < 0.1 or i == self.max_iterations - 1:
                return gen[ind].clone().detach(), zhat[ind].clone().detach()

            print("----")
            print("Norms", best_l2_loss.item())
            print("Losses", best_adv_loss.item())
            print("Success rate: ", 1. - not_dones_mask.mean())
            print("Lambda: ", LAM[:5])

            ((total_loss * not_dones_mask).mean() / not_dones_mask.mean()).backward()
            opt.step()
            lam_opt.step()
            # LAM.data.clamp_(min=0.0)
            lr_maker.step()










