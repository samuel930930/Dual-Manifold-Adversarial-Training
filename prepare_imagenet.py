import os
import glob
import argparse
import numpy as np
from collections import defaultdict
import torch
import torch.backends.cudnn as cudnn
from torchvision.utils import save_image
import h5py

from stylegan.stylegan_generator_model import StyleGANGeneratorModel

parser = argparse.ArgumentParser(description="Prepare manifold dataset")
parser.add_argument("--debug", action='store_true')
args = parser.parse_args()

DEBUG = args.debug

gan_path = './stylegan/pretrain/stylegan_imagenet.pth'
latents_dir = 'stylegan/output/latents'
train_dir = 'stylegan/dataset/train'
test_dir = 'stylegan/dataset/test'
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

latents_pt_files = sorted(glob.glob(os.path.join(latents_dir, '*.pt')))
if DEBUG:
    latents_pt_files = latents_pt_files[:1000]
print(f'Found {len(latents_pt_files)} latents files')

dataset_summary = defaultdict(list)
training_set_partition = {}

cnt = 0
for latents_pt in latents_pt_files:
    ckpt = torch.load(latents_pt)
    label = int(ckpt['y'].numpy())
    dataset_summary[label].append(latents_pt.split('/')[-1])
    print(label, cnt)
    cnt += 1

print('------ Dataset Summary ------')
for k, v in dataset_summary.items():
    print(f'class {k} has {len(v)} images')
    training_set_partition[k] = int(len(v) * 0.9)


# Load StyleGAN
cudnn.benchmark = True
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
    for k, v in dataset_summary.items():
        trainset_filename = os.path.join(train_dir, f'{k}.h5')
        testset_filename = os.path.join(test_dir, f'{k}.h5')

        train_size = training_set_partition[k] if not DEBUG else 10
        test_size = len(v) - train_size if not DEBUG else 10

        images = []
        latents = []
        for train_pt in v[:train_size]:
            ckpt = torch.load(os.path.join(latents_dir, train_pt))
            z = ckpt['z'].cuda()
            img = gen.synthesis(z)
            images.append(img.cpu().numpy())
            latents.append(z.cpu().numpy())

            if DEBUG or len(images) % 100 == 0:
                print(f'[{len(images)}/{train_size}]')
                save_image(
                    img,
                    os.path.join(train_dir, f'sanity_{k}_{len(images)}.png'),
                    nrow=1,
                    normalize=True,
                    range=(-1., 1.)
                )

        images = np.concatenate(images, axis=0)
        latents = np.concatenate(latents, axis=0)

        print(images.shape)
        print(latents.shape)

        # create hdf5
        with h5py.File(trainset_filename, 'w') as hf:
            hf.create_dataset(
                'images',
                data=images,
                dtype=np.float32
            )
            hf.create_dataset(
                'latents',
                data=latents,
                dtype=np.float32
            )

        images = []
        latents = []
        for test_pt in v[-test_size:]:
            ckpt = torch.load(os.path.join(latents_dir, test_pt))
            z = ckpt['z'].cuda()
            img = gen.synthesis(z)
            images.append(img.cpu().numpy())
            latents.append(z.cpu().numpy())

            if DEBUG or len(images) % 100 == 0:
                print(f'[{len(images)}/{test_size}]')
                save_image(
                    img,
                    os.path.join(test_dir, f'sanity_{k}_{len(images)}.png'),
                    nrow=1,
                    normalize=True,
                    range=(-1., 1.)
                )

        images = np.concatenate(images, axis=0)
        latents = np.concatenate(latents, axis=0)

        with h5py.File(testset_filename, 'w') as hf:
            hf.create_dataset(
                'images',
                data=images,
                dtype=np.float32
            )
            hf.create_dataset(
                'latents',
                data=latents,
                dtype=np.float32
            )







