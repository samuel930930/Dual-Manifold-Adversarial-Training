import os
import torch
import torch.nn as nn
import time
from utils.loaders import get_optimizer, get_scheduler, get_generator, get_transform
from utils.losses import PerceptualLoss, per_pixel_l2_dist
from stylegan_old.stylegan_generator_model import StyleGANGeneratorModel
import math
from torch import optim
# (0) see how can do better for rec_rr=1
# (1) smoothed defense. target = x + epsilon
lpips = PerceptualLoss().cuda()
norm = torch.nn.L1Loss(reduction='none').cuda()

def get_lr(t, initial_lr, rampdown=0.25, rampup=0.05):
    lr_ramp = min(1, (1 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * min(1, t / rampup)

    return initial_lr * lr_ramp

def latent_noise(latent, strength):
    noise = torch.randn_like(latent) * strength

    return latent + noise

class DefenseGANProjection(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, z_tiled, gan, rec_iters, rec_rr, w_lpips, optimizer, schedular, latent_std):
        # we assume z is already tiled based on rec_rr
        assert x.size(0) * rec_rr == z_tiled.size(0)

        # important !!! avoid modifying the gradient of input in-place
        x = x.detach().clone()
        batch_size = x.size(0)
        x_tiled = torch.repeat_interleave(x, rec_rr, dim=0)
        # noise = torch.randn_like(x_tiled) * 0.5
        # x_tiled = x_tiled + noise

        start_time = time.time()
        with torch.enable_grad():
            for i in range(rec_iters):
                t = i / rec_iters
                lr = get_lr(t, 0.1)
                optimizer.param_groups[0]['lr'] = lr
                noise_strength = latent_std * 0.05 * max(0, 1 - t / 0.75) ** 2
                latent_n = latent_noise(z_tiled, noise_strength.item())
                img_gen = gan.synthesis(latent_n)

                p_loss = lpips(img_gen, x_tiled).sum()
                norm_loss = norm(img_gen, x_tiled).mean(dim=(1, 2, 3)).sum()
                # --- IMPORTANT ---
                # different from normal network training where a fixed set of parameters are updated,
                # in this case, each instance in a batch has its own parameters to update,
                # therefore, the total loss should not be averaged over number of instances
                loss = w_lpips * p_loss + norm_loss
                loss_logging = per_pixel_l2_dist(img_gen, x_tiled)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (i + 1) % 50 == 0:
                    print(f'===> Iter: {i+1} | '
                          f'PP-L2: {loss_logging.item():.6f} | '
                          f'Perceptual: {p_loss.mean().item():.6f} | '
                          f'Time: {time.time()-start_time:.3f}')
                    start_time = time.time()

                schedular.step()

        gen = gan.synthesis(z_tiled)
        loss_tiled = (gen - x_tiled).pow(2).mean(dim=(1, 2, 3)) + w_lpips * lpips(gen, x_tiled)
        loss_tiled = loss_tiled.view(-1, rec_rr)  # (B, r)
        indices = torch.argmin(loss_tiled, dim=1)
        offsets = torch.arange(0, batch_size).cuda() * rec_rr

        return gen[indices + offsets].detach().clone(), z_tiled[indices + offsets].detach().clone()

    @staticmethod
    def backward(ctx, grad_output, grad_z):
        grad_input = grad_output.clone()
        return grad_input, None, None, None, None, None, None, None


class Preprocessor(nn.Module):
    def __init__(self, cfg):
        super(Preprocessor, self).__init__()
        self.batch_size = cfg.dataset.test.batch_size
        gan_path = '/vulcanscratch/cplau/Code/Python/InvGAN-PyTorch-master/stylegan_old/pretrain/stylegan_imagenet.pth'
        # load generator
        gen = StyleGANGeneratorModel()
        state_dict = torch.load(gan_path)
        var_name = 'truncation.truncation'
        state_dict[var_name] = gen.state_dict()[var_name]
        gen.load_state_dict(state_dict)

        for p in gen.parameters():
            p.requires_grad_(False)

        gen.cuda()
        gen.eval()
        self.gan = gen

        with torch.no_grad():
            self.noise_sample = torch.randn(10000, 512).cuda()
            self.latent_out = self.gan.mapping(self.noise_sample)
            self.latent_mean = self.latent_out.mean(0)
            self.latent_std = ((self.latent_out - self.latent_mean).pow(2).sum() / 10000) ** 0.5
            print('Finished estimating w statistics ..')

        # load transformation
        transform = get_transform(cfg)
        # self.gan = nn.Sequential(self.gan, transform.gan_deprocess_layer)

        defense_cfg = cfg.defense.args
        self.optimizer_fn = get_optimizer(defense_cfg.optimizer)
        self.scheduler_fn = get_scheduler(defense_cfg.scheduler)
        self.rec_rr = defense_cfg.rec_rr
        self.rec_iters = defense_cfg.rec_iters
        self.w_lpips = defense_cfg.w_lpips
        self.return_latents = True

    def forward(self, x):
        # z = torch.randn(x.size(0)*self.rec_rr, self.z_dim, requires_grad=True, device=x.device)
        z = self.latent_mean.detach().clone().view(-1, 1, 512).repeat(x.size(0), 14, 1)
        z.requires_grad = True
        optimizer = self.optimizer_fn(params=[z])
        schedular = self.scheduler_fn(optimizer=optimizer)

        output = DefenseGANProjection.apply(x,
                                          z,
                                          self.gan,
                                          self.rec_iters,
                                          self.rec_rr,
                                          self.w_lpips,
                                          optimizer,
                                          schedular,
                                          self.latent_std)
        return output if self.return_latents else output[0]





