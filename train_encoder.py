import os
import argparse
import shutil
import time
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.backends.cudnn as cudnn
from utils.loaders import \
    set_random_seed, load_config, get_generator, \
    get_transform, get_encoder, get_discriminator, get_optimizer
from utils.losses import GANLoss, RecLoss
from torchvision.utils import save_image

# parse command line options
parser = argparse.ArgumentParser(description="Train Encoder")
parser.add_argument("--config", default="experiments/encoders/mnist.yml", help="config file")
args = parser.parse_args()

cfg = load_config(args.config)
dataset_cfg = cfg.dataset.train
use_gan_loss = (cfg.train.adv_loss.type != 'none')
print(cfg)

# set paths
output_dir = cfg.encoder.path
log_dir = os.path.join(output_dir, 'log')
vis_dir = os.path.join(output_dir, 'vis')
checkpoint_dir = os.path.join(output_dir, 'checkpoint')

os.makedirs(log_dir, exist_ok=True)
os.makedirs(vis_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)

# set random seed
set_random_seed(cfg)
cudnn.benchmark = True

writer = SummaryWriter(log_dir=log_dir)
# load data transformation
transform = get_transform(cfg)
# load generator
netG = get_generator(cfg, cfg.generator)
state_dict = torch.load(cfg.generator.ckpt)
netG.load_state_dict(state_dict['netG'])
netG = torch.nn.Sequential(netG, transform.gan_deprocess_layer)
netG.cuda()
netG.eval()

netE = get_encoder(cfg, cfg.encoder)
netE.cuda()
netE.train()

# set optimizers
optE = get_optimizer(cfg.optimizer.encoder)(params=netE.parameters())
# set loss function
image_criterion = RecLoss(cfg.train.image_loss.type)
latent_criterion = RecLoss(cfg.train.latent_loss.type)
# set weights
w_image = cfg.train.image_loss.weight
w_latent = cfg.train.latent_loss.weight


if use_gan_loss:
    netD = get_discriminator(cfg, cfg.discriminator)
    netD.cuda()
    netD.train()
    optD = get_optimizer(cfg.optimizer.discriminator)(params=netD.parameters())
    gan_criterion = GANLoss(gan_mode=cfg.train.adv_loss.type)
    w_adv = cfg.train.adv_loss.weight

for iters in range(cfg.max_iters):
    start_time = time.time()

    if use_gan_loss:
        optD.zero_grad()
        z = torch.randn(dataset_cfg.batch_size, cfg.generator.args.z_dim).cuda()

        x_real = netG(z).detach()

        if cfg.train.denoise:
            noise = torch.randn_like(x_real) * 0.1
            x_in = x_real + noise
        else:
            x_in = x_real

        z_hat = netE(x_in)
        x_fake = netG(z_hat).detach()

        real_pred = netD(x_real)
        fake_pred = netD(x_fake)

        d_real_loss = gan_criterion(real_pred, target_is_real=True, is_disc=True)
        d_fake_loss = gan_criterion(fake_pred, target_is_real=False, is_disc=True)
        d_loss = d_real_loss + d_fake_loss
        d_loss.backward()
        optD.step()

    optE.zero_grad()
    z = torch.randn(dataset_cfg.batch_size, cfg.generator.args.z_dim).cuda()
    x_real = netG(z).detach()
    z_hat = netE(x_real)
    x_fake = netG(z_hat)
    latent_loss = latent_criterion(z_hat, z)
    image_loss = image_criterion(x_fake, x_real)

    e_loss = w_image * image_loss + w_latent * latent_loss

    if use_gan_loss:
        fake_pred = netD(x_fake)
        e_adv_loss = gan_criterion(fake_pred, target_is_real=True)
        e_loss += w_adv * e_adv_loss

    e_loss.backward()
    optE.step()

    if iters % cfg.log_iters == 0:
        print("{} secs/iter".format(time.time() - start_time))
        writer.add_scalar('Loss/E_loss/image_l2', image_loss.item(), iters)
        writer.add_scalar('Loss/E_loss/latent_l2', latent_loss.item(), iters)
        if use_gan_loss:
            writer.add_scalar('Loss/D_loss', d_loss.item(), iters)
            writer.add_scalar('Loss/E_loss/adv', e_adv_loss.item(), iters)

    if iters % cfg.visualize_iters == 0:
        x_real_path = os.path.join(vis_dir, f'iter_{iters}_real.png')
        x_fake_path = os.path.join(vis_dir, f'iter_{iters}_fake.png')
        save_image(x_in, x_real_path, nrow=8, padding=2)
        save_image(x_fake, x_fake_path, nrow=8, padding=2)

    if iters % cfg.save_iters == 0:
        filename = 'iter_{0:06d}.pt'.format(iters)
        latest_filename = 'latest.pt'

        path = os.path.join(checkpoint_dir, filename)
        latest_path = os.path.join(checkpoint_dir, latest_filename)

        torch.save({'netE': netE.state_dict()}, path)
        shutil.copyfile(path, latest_path)

