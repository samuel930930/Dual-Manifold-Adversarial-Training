import torch
from robustness import datasets
from robustness.tools.helpers import get_label_mapping
from torchvision import transforms


class CustomImageNet(datasets.DataSet):
    '''
    ImageNet Dataset [DDS+09]_.

    Requires ImageNet in ImageFolder-readable format.
    ImageNet can be downloaded from http://www.image-net.org. See
    `here <https://pytorch.org/docs/master/torchvision/datasets.html#torchvision.datasets.ImageFolder>`_
    for more information about the format.

    .. [DDS+09] Deng, J., Dong, W., Socher, R., Li, L., Li, K., & Fei-Fei, L. (2009). ImageNet: A large-scale hierarchical image database. 2009 IEEE Conference on Computer Vision and Pattern Recognition, 248-255.

    '''
    def __init__(self, data_path, custom_grouping, **kwargs):
        """
        """
        ds_name = 'custom_imagenet'
        ds_kwargs = {
            'num_classes': len(custom_grouping),
            'mean': torch.tensor([0.4717, 0.4499, 0.3837]),
            'std': torch.tensor([0.2600, 0.2516, 0.2575]),
            'custom_class': None,
            'label_mapping': get_label_mapping(ds_name,
                custom_grouping),
            'transform_train': TRAIN_TRANSFORMS_IMAGENET,
            'transform_test': TEST_TRANSFORMS_IMAGENET
        }
        super(CustomImageNet, self).__init__(ds_name,
                data_path, **ds_kwargs)

TRAIN_TRANSFORMS_IMAGENET = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
TEST_TRANSFORMS_IMAGENET = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
