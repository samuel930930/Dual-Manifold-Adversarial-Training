import os
import argparse
import time
import shutil
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils.loaders import \
    set_random_seed, load_config, get_generator, \
    get_discriminator, get_optimizer, get_dataset, get_transform
from utils.losses import GANLoss
from torchvision.utils import save_image

# parse command line options
parser = argparse.ArgumentParser(description="Train GAN")
parser.add_argument("--config", default="experiments/gans/cifar.yml", help="config file")
args = parser.parse_args()

cfg = load_config(args.config)
dataset_cfg = cfg.dataset.train
train_cfg = cfg.train
print(cfg)

# set paths (tensorboard loggers, checkpoint)
os.makedirs(cfg.dataset.train.path, exist_ok=True)
os.makedirs(cfg.dataset.test.path, exist_ok=True)

output_dir = cfg.generator.path
log_dir = os.path.join(output_dir, 'log')
vis_dir = os.path.join(output_dir, 'vis')
checkpoint_dir = os.path.join(output_dir, 'checkpoint')

os.makedirs(log_dir, exist_ok=True)
os.makedirs(vis_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)

# set random seed
set_random_seed(cfg)
cudnn.benchmark = True

# set logger
writer = SummaryWriter(log_dir=log_dir)

# set models
netG = get_generator(cfg, cfg.generator)
netD = get_discriminator(cfg, cfg.discriminator)
netG.cuda(), netD.cuda()
netG.train(), netD.train()
print(netG)
print(netD)

# set optimizers
optG = get_optimizer(cfg.optimizer.generator)(params=netG.parameters())
optD = get_optimizer(cfg.optimizer.discriminator)(params=netD.parameters())

# set dataset, dataloader
dataset = get_dataset(cfg)
transform = get_transform(cfg)
trainset = dataset(root=dataset_cfg.path, train=True, transform=transform.gan_training)
trainloader = DataLoader(trainset, batch_size=dataset_cfg.batch_size, num_workers=0, shuffle=True)

# set gan loss
gan_loss = GANLoss(gan_mode=train_cfg.loss_type)

# training, visualizing, saving
iters = 0
for epoch in range(cfg.num_epochs):
    for i, data in enumerate(trainloader):
        start_time = time.time()
        real_imgs = data[0].cuda()

        # update discriminator
        for _ in range(train_cfg.args.n_dis):
            optD.zero_grad()
            z = torch.randn(real_imgs.size(0), cfg.generator.args.z_dim).cuda()
            gen_imgs = netG(z).detach()
            real_pred = netD(real_imgs)
            fake_pred = netD(gen_imgs)

            d_real_loss = gan_loss(real_pred, target_is_real=True, is_disc=True)
            d_fake_loss = gan_loss(fake_pred, target_is_real=False, is_disc=True)
            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            optD.step()

        # update generator
        optG.zero_grad()
        z = torch.randn(real_imgs.size(0), cfg.generator.args.z_dim).cuda()
        gen_imgs = netG(z)
        fake_pred = netD(gen_imgs)

        g_loss = gan_loss(fake_pred, target_is_real=True)
        g_loss.backward()
        optG.step()

        iters += 1
        if iters % cfg.log_iters == 0:
            print("epoch: {}, {} secs/iter".format(epoch, time.time() - start_time))
            writer.add_scalar('Loss/D_loss', d_loss.item(), iters)
            writer.add_scalar('Loss/G_loss', g_loss.item(), iters)

        if iters % cfg.visualize_iters == 0:
            real_image_path = os.path.join(vis_dir, 'real.png')
            image_path = os.path.join(vis_dir, f'iter_{iters}.png')
            save_image(real_imgs, real_image_path,
                       nrow=8, padding=2, normalize=True, range=(-1.0, 1.0))
            save_image(gen_imgs, image_path,
                       nrow=8, padding=2, normalize=True, range=(-1.0, 1.0))

        if iters % cfg.save_iters == 0:
            filename = 'iter_{0:06d}.pt'.format(iters)
            latest_filename = 'latest.pt'

            path = os.path.join(checkpoint_dir, filename)
            latest_path = os.path.join(checkpoint_dir, latest_filename)

            torch.save({'netG': netG.state_dict(),
                        'netD': netD.state_dict()},
                       path)
            shutil.copyfile(path, latest_path)

    if iters >= cfg.max_iters:
        break

