import os
import logging
from tqdm import tqdm
import argparse
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from utils.loaders import \
    set_device, move_to_device, set_random_seed, load_config, get_optimizer, \
    get_scheduler, get_classifier, get_dataset, get_transform, get_attack
from utils.metrics import AverageMeter
from stylegan.stylegan_generator_model import StyleGANGeneratorModel
from advertorch.attacks import PGDAttack

# parse command line options
parser = argparse.ArgumentParser(description="On-manifold adv training")
parser.add_argument("--config", default="experiments/classifiers/fimagenet_style_pgd8.yml")
args = parser.parse_args()

cfg = load_config(args.config)
trainset_cfg = cfg.dataset.train
testset_cfg = cfg.dataset.test
print(cfg)

output_dir = cfg.classifier.path
os.makedirs(output_dir, exist_ok=True)
checkpoint_path = os.path.join(output_dir, 'classifier.pt')
output_filename = os.path.join(output_dir, 'metrics.txt')
logging.basicConfig(level=logging.INFO, filename=output_filename, filemode='w')

# set device and random seed
set_device(cfg)
set_random_seed(cfg)
cudnn.benchmark = True

# set classifier
net = get_classifier(cfg, cfg.classifier)
state_dict = torch.load(checkpoint_path)['state_dict']
net.load_state_dict(state_dict)
for p in net.parameters():
    p.requires_grad_(False)
net.cuda()
net.eval()

# load GAN
gan = StyleGANGeneratorModel()
state_dict = torch.load('./stylegan/pretrain/stylegan_imagenet.pth')
var_name = 'truncation.truncation'
state_dict[var_name] = gan.state_dict()[var_name]
gan.load_state_dict(state_dict)
# reduce memory consumption
gan = gan.synthesis
for p in gan.parameters():
    p.requires_grad_(False)
gan.cuda()
gan.eval()

model = torch.nn.Sequential(gan, net)

pgd_iters = [10, 50]
test_attacker = PGDAttack(predict=net,
                          eps=8.0/255,
                          eps_iter=2.0/255,
                          nb_iter=50,
                          clip_min=-1.0,
                          clip_max=1.0)

# set dataset, dataloader
dataset = get_dataset(cfg)
transform = get_transform(cfg)
testset = dataset(root=testset_cfg.path, train=False)
testloader = DataLoader(testset,
                        batch_size=10,
                        num_workers=4,
                        shuffle=False)


for nb_iter in pgd_iters:
    acc_clean = AverageMeter()
    acc_adv_image = AverageMeter()
    acc_adv_latent = AverageMeter()

    progress_bar = tqdm(testloader)

    attacker = PGDAttack(
        predict=net,
        eps=8.0 / 255,
        eps_iter=2.0 / 255,
        nb_iter=nb_iter,
        clip_min=-1.0,
        clip_max=1.0)

    for images, _, labels in progress_bar:
        images, labels = images.cuda(), labels.cuda()

        images_adv = attacker.perturb(images, labels)

        with torch.no_grad():
            pred_clean = net(images).argmax(dim=1)
            pred_adv = net(images_adv).argmax(dim=1)
            acc_clean.update((pred_clean == labels).float().mean().item() * 100.0, images.size(0))
            acc_adv_image.update((pred_adv == labels).float().mean().item() * 100.0, images.size(0))

        progress_bar.set_description(
            'Clean: {acc_clean.avg:.3f} '
            'PGD-{nb_iter}-image: {acc_adv_image.avg:.3f}'.format(acc_clean=acc_clean, nb_iter=nb_iter, acc_adv_image=acc_adv_image)
        )
    logging.info(f'Clean {acc_clean.avg:3f}')
    logging.info(f'PGD-{nb_iter}-image {acc_adv_image.avg:3f}')

    progress_bar = tqdm(testloader)
    attacker = PGDAttack(
        predict=model,
        eps=0.02,
        eps_iter=0.005,
        nb_iter=nb_iter,
        clip_min=None,
        clip_max=None)

    for images, latents, labels in progress_bar:
        images, latents, labels = images.cuda(), latents.cuda(), labels.cuda()

        latents_adv = attacker.perturb(latents, labels)
        images_adv = gan(latents_adv)

        with torch.no_grad():
            pred_adv = net(images_adv).argmax(dim=1)
            acc_adv_latent.update((pred_adv == labels).float().mean().item() * 100.0, images.size(0))

        progress_bar.set_description(
            f'PGD-{nb_iter}-latent: {acc_adv_latent.avg:.3f} '
        )
    logging.info(f'PGD-{nb_iter}-latent {acc_adv_latent.avg:3f}')



