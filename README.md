# PyTorch Implementation of Dual Manifold Adversarial Training (DMAT)

## Requirements
- CUDA 10.0.130 or newer, cudnn v7.6.5 or newer.
- Python 3.6.
- PyTorch 1.3.1 or newer with GPU support.
- tqdm
- advertorch (`pip install advertorch`)

## Dataset and trained StyleGAN
You can download the dataset and trained StyleGAN by the [link](https://drive.google.com/drive/folders/1DRsiquKEClTaje2nXXGugOiqfGICkv0-?usp=sharing).

## Structure of the configs
Since we need to handle GAN, classifier training, and adversarial attack in the same place, it's important to have a structured configuration. Take classifier training as an example, in `experiments/encoders/mnist.yml`:
```
num_epochs: 20

classifier:
  name: imagenet_classifier
  path: runs/classifiers/mixed10_pgd5_pgd5_sgd_sam

optimizer:
  module: torch.optim
  name: SGD
  args:
    lr: 0.1
    momentum: 0.9
    weight_decay: 0.0001

scheduler:
  type: cyclic
  args:
    lr_epochs: !!python/tuple [0,1,6,12,15,20]
    lr_values: !!python/tuple [0,0.4,0.04,0.004,0.0004,0.0001]

image_attack:
  module: advertorch.attacks
  name: PGDAttack
  args:
    eps: 0.03137255 # 8/255 in [-1, 1]
    eps_iter: 0.00784314 # 2/255 in [-1, 1]
    nb_iter: 5
    clip_min: -1.0
    clip_max: 1.0

latent_attack:
  module: advertorch.attacks
  name: PGDAttack
  args:
    eps: 0.02
    eps_iter: 0.005
    nb_iter: 5
    clip_min: null
    clip_max: null

dataset:
  name: mixed10
  num_classes: 10
  train:
    path: /vulcanscratch/cplau/Code/Python/InvGAN-PyTorch-master/stylegan_old/dataset
    batch_size: 32
  test:
    path: /vulcanscratch/cplau/Code/Python/InvGAN-PyTorch-master/stylegan_old/dataset
    batch_size: 32

...
```
- `num_epoch`: Number of epoch
 - `classifier.path`: the directory that the training results are saved, which include checkpoints, and images.
 - `classifier.name`: the .py file that the generator architecture is defined. Since it is a classifier, the .py file is assumed to be under `classifier`.
 - `optimizer`: Setting of optimizer
 - `image_attack`: Setting of adversarial attack in image space
 - `image_attack.name`: Type of attack
 - `image_attack.args`: Include eps (attack budget), eps_iter (step size of each iteration), nb_iter (number of iteration), clip (range of the perturbed image)
 - `latent_attack`: Setting of adversarial attack in latent space
 - `dataset`: Setting of dataset

If we want to perform experiments to find proper hyperparameters (e.g. sweeping over different learning rates). It's better to create a new config file for each hyperparameter. 

## Preparing Classifiers
We train classifiers from scratch for OM_ImageNet.
```
> python prepare_classifiers_mix10.py --config experiments/classifiers/mixed10_adv_pgd5_pgd5_sgd_sam_clean.yml
```
## Evaluation
After we trained the classifier, we can use various attacks to evaluate the classifier. See the scripts in `script`.



