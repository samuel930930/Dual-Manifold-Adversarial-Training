import os
import torch
import glob
import h5py
import torch.utils.data as data
import torchvision.transforms as transforms
from advertorch.utils import NormalizeByChannelMeanStd

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

MEAN = (2*0.485-1, 2*0.456-1, 2*0.406-1)
STD = (2*0.229, 2*0.224, 2*0.225)


class Transform(object):
    classifier_training = transforms.Compose([
        transforms.Normalize((-1., -1., -1.), (2., 2., 2.)),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])
    classifier_testing = transforms.Compose([
        transforms.Normalize((-1., -1., -1.), (2., 2., 2.)),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])

    default = transforms.ToTensor()
    gan_deprocess_layer = []
    classifier_preprocess_layer = NormalizeByChannelMeanStd(MEAN, STD).cuda()


class Dataset(data.Dataset):
    def __init__(self, root, train=False, transforms=None, transform=None, target_transform=None):

        self.transforms = transforms
        self.transform = transform
        self.target_transform = target_transform

        if train:
            self.db_paths = glob.glob(os.path.join(root, 'train', '*.h5'))
            split = 'train'
        else:
            self.db_paths = glob.glob(os.path.join(root, 'test', '*.h5'))
            split = 'test'

        self.targets = []
        self.indices = []
        count = 0
        for db_path in self.db_paths:
            with h5py.File(db_path, 'r') as h_file:
                count += h_file['images'].shape[0]
                self.indices.append(count)
                self.targets.append(int(os.path.split(db_path)[1][0]))

        self.length = count

        print(f'{split} dataset size: {count}')

    def __getitem__(self, index):
        db_ind = 0
        offset = 0

        for ind in self.indices:
            if index < ind:
                break
            db_ind += 1
            offset = ind

        db_path = self.db_paths[db_ind]
        index = index - offset
        with h5py.File(db_path, 'r') as h_file:
            image = h_file['images'][index, :, :, :]
            latent = h_file['latents'][index, :, :]

        target = self.targets[db_ind]

        # assume images in [-1, 1]
        image = torch.tensor(image).float()
        latent = torch.tensor(latent).float()
        target = torch.tensor(target).long()

        return image, latent, target

    def __len__(self):
        return self.length

