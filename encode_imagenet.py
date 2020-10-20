import os
import math
import time
import argparse
import torch
import torch.backends.cudnn as cudnn
from torch import optim
import torch.nn.functional as F
from torchvision.utils import save_image
from stylegan.stylegan_generator_model import StyleGANGeneratorModel
from utils.losses import PerceptualLoss

import numpy as np
import random

from datasets.imagenet import CustomImageNet
from robustness.tools.imagenet_helpers import common_superclass_wnid, ImageNetHierarchy


parser = argparse.ArgumentParser(description="StyleGAN Projector")
parser.add_argument("--w_norm", type=float, default=100)
parser.add_argument("--batch_size", type=int, default=10)
parser.add_argument("--world_size", type=int, default=1)
parser.add_argument("--rank", type=int, default=0)
args = parser.parse_args()


def get_lr(t, initial_lr, rampdown=0.25, rampup=0.05):
    lr_ramp = min(1, (1 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * min(1, t / rampup)

    return initial_lr * lr_ramp


def latent_noise(latent, strength):
    noise = torch.randn_like(latent) * strength

    return latent + noise

# ------ SET PATHS TO IMAGENET ------
in_path = '/vulcan/scratch/andylin/imagenet/ImageNet/data'
in_info_path = '/vulcan/scratch/andylin/imagenet/ImageNet/info'
# -----------------------------------

gan_path = './stylegan/pretrain/stylegan_imagenet.pth'
output_dir = './stylegan/output'
latents_dir = os.path.join(output_dir, 'latents')
output_dir = os.path.join(output_dir, str(args.w_norm))

os.makedirs(output_dir, exist_ok=True)
os.makedirs(latents_dir, exist_ok=True)

# parameters
n_mean_latent = 10000
base_lr = 0.1
step = 1000
noise = 0.05
noise_ramp = 0.75

# random seed
seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)
cudnn.benchmark = True

# Set losses
lpips = PerceptualLoss().cuda()
norm = torch.nn.L1Loss(reduction='none').cuda()

# Load StyleGAN
gen = StyleGANGeneratorModel()
state_dict = torch.load(gan_path)
var_name = 'truncation.truncation'
state_dict[var_name] = gen.state_dict()[var_name]
gen.load_state_dict(state_dict)

for p in gen.parameters():
    p.requires_grad_(False)

gen.cuda()
gen.eval()

with torch.no_grad():
    noise_sample = torch.randn(n_mean_latent, 512).cuda()
    latent_out = gen.mapping(noise_sample)

    latent_mean = latent_out.mean(0)
    latent_std = ((latent_out - latent_mean).pow(2).sum() / n_mean_latent) ** 0.5
    print('Finished estimating w statistics ..')

# Define dataset (install robustness package, set imagenet path)
in_hier = ImageNetHierarchy(in_path, in_info_path)
superclass_wnid = common_superclass_wnid('mixed_10')
class_ranges, label_map = in_hier.get_subclasses(superclass_wnid, balanced=True)
custom_dataset = CustomImageNet(in_path, class_ranges)
train_loader, test_loader = custom_dataset.make_loaders(workers=4,
                                                        batch_size=args.batch_size,
                                                        shuffle_train=False,
                                                        shuffle_val=False)

n_batch = len(train_loader)
chunk_size = n_batch // args.world_size
start_ind = args.rank * chunk_size
end_ind = (args.rank + 1) * chunk_size
image_ind = 0

for n, (images, labels) in enumerate(train_loader):
    sz = images.size(0)

    if n < start_ind or n >= end_ind:
        image_ind += sz
        continue

    skip = True
    for b in range(sz):
        latents_filename = os.path.join(latents_dir, f'{image_ind+b:07d}.pt')
        # if any image in this batch is not done yet, process the entire batch
        if not os.path.isfile(latents_filename):
            skip = False

    if skip:
        image_ind += sz
        continue

    images = images.cuda()
    # scale to [-1, 1]
    images = 2 * images - 1

    # initialize w
    latent_in = latent_mean.detach().clone().view(-1, 1, 512).repeat(sz, 14, 1)
    latent_in.requires_grad = True

    # optimizer
    optimizer = optim.Adam([latent_in], lr=base_lr)

    start_time = time.time()
    for i in range(step):
        t = i / step
        lr = get_lr(t, base_lr)
        optimizer.param_groups[0]['lr'] = lr
        noise_strength = latent_std * noise * max(0, 1 - t / noise_ramp) ** 2
        latent_n = latent_noise(latent_in, noise_strength.item())
        img_gen = gen.synthesis(latent_n)

        p_loss = lpips(img_gen, images).sum()
        norm_loss = norm(img_gen, images).mean(dim=(1, 2, 3)).sum()

        loss = p_loss + args.w_norm * norm_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 50 == 0:
            print(f'[{n}/{n_batch}][{i}/{step}] perceptual: {p_loss.item():.4f}; '
                  f'norm: {norm_loss.item():.4f}; lr: {lr:.4f}')
    print(f'------ {time.time() - start_time} sec per batch ------')
    save_image(img_gen,
               os.path.join(output_dir, 'rec_{:04d}.png'.format(n)),
               nrow=1,
               normalize=True,
               range=(-1., 1.))
    save_image(images,
               os.path.join(output_dir, 'orig_{:04d}.png'.format(n)),
               nrow=1,
               normalize=True,
               range=(-1., 1.))

    for b in range(sz):
        latents_filename = os.path.join(latents_dir, f'{image_ind:07d}.pt')
        torch.save({
            'z': latent_in[b:b+1].cpu(),
            'y': labels[b:b+1]
        }, latents_filename)
        image_ind += 1
