import os
import torch
import torch.nn as nn
import time
from utils.loaders import get_optimizer, get_scheduler, get_generator, get_transform
from utils.losses import per_pixel_l2_dist


class DefenseGANProjection(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, z_tiled, gan, rec_iters, rec_rr, optimizer, scheduler):
        # we assume z is already tiled based on rec_rr
        assert x.size(0) * rec_rr == z_tiled.size(0)

        # important !!! avoid modifying the gradient of input in-place
        x = x.detach().clone()
        batch_size = x.size(0)
        x_tiled = torch.repeat_interleave(x, rec_rr, dim=0)

        start_time = time.time()
        with torch.enable_grad():
            for i in range(rec_iters):
                optimizer.zero_grad()
                gen = gan(z_tiled)
                loss_tiled = (gen - x_tiled).pow(2).mean(dim=(1, 2, 3))
                # --- IMPORTANT ---
                # different from normal network training where a fixed set of parameters are updated,
                # in this case, each instance in a batch has its own parameters to update,
                # therefore, the total loss should not be averaged over number of instances
                loss = loss_tiled.sum()
                loss_logging = per_pixel_l2_dist(gen, x_tiled)
                loss.backward()
                optimizer.step()

                if (i + 1) % 50 == 0:
                    print(f'===> Iter: {i+1} | '
                          f'PP-L2: {loss_logging.item():.6f} | '
                          f'Time: {time.time()-start_time:.3f}')
                    start_time = time.time()

                scheduler.step()

        gen = gan(z_tiled)
        loss_tiled = (gen - x_tiled).pow(2).mean(dim=(1, 2, 3))
        loss_tiled = loss_tiled.view(-1, rec_rr)  # (B, r)
        indices = torch.argmin(loss_tiled, dim=1)
        offsets = torch.arange(0, batch_size).cuda() * rec_rr

        return gen[indices + offsets].detach().clone(), z_tiled[indices + offsets].detach().clone()

    @staticmethod
    def backward(ctx, grad_output, grad_z):
        grad_input = grad_output.clone()
        return grad_input, None, None, None, None, None, None


class Preprocessor(nn.Module):
    def __init__(self, cfg):
        super(Preprocessor, self).__init__()

        # load generator
        self.gan = get_generator(cfg, cfg.generator)
        self.z_dim = cfg.generator.args.z_dim
        state_dict = torch.load(cfg.generator.ckpt)
        self.gan.load_state_dict(state_dict['netG'])
        for p in self.gan.parameters():
            p.requires_grad_(False)

        # load transformation
        transform = get_transform(cfg)
        self.gan = nn.Sequential(self.gan, transform.gan_deprocess_layer)

        defense_cfg = cfg.defense.args
        self.optimizer_fn = get_optimizer(defense_cfg.optimizer)
        self.scheduler_fn = get_scheduler(defense_cfg.scheduler)
        self.rec_rr = defense_cfg.rec_rr
        self.rec_iters = defense_cfg.rec_iters
        self.return_latents = True

    def forward(self, x):
        z = torch.randn(x.size(0)*self.rec_rr, self.z_dim, requires_grad=True, device=x.device)
        optimizer = self.optimizer_fn(params=[z])
        scheduler = self.scheduler_fn(optimizer=optimizer)

        output = DefenseGANProjection.apply(x,
                                          z,
                                          self.gan,
                                          self.rec_iters,
                                          self.rec_rr,
                                          optimizer,
                                          scheduler)

        return output if self.return_latents else output[0]





