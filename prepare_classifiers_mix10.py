import os
import shutil
import logging
from tqdm import tqdm
import numpy as np
import argparse
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from datasets.mixed10_natural import Dataset as test_dataset
from torch.utils.data.distributed import DistributedSampler
from utils.loaders import \
    set_device, move_to_device, set_random_seed, load_config, load_optimizer, \
    get_classifier, get_dataset, get_transform, get_attack
from utils.metrics import AverageMeter
from stylegan_old.stylegan_generator_model import StyleGANGeneratorModel
from advertorch.attacks import PGDAttack
from advertorch.context import ctx_noparamgrad_and_eval

# parse command line options
parser = argparse.ArgumentParser(description="On-manifold adv training")
parser.add_argument("--config", default="experiments/classifiers/mixed10_adv_pgd5_pgd5_sgd_sam.yml")
parser.add_argument("--resume", default="")
args = parser.parse_args()

cfg = load_config(args.config)
trainset_cfg = cfg.dataset.train
testset_cfg = cfg.dataset.test
print(cfg)

output_dir = cfg.classifier.path + "_clean"
os.makedirs(output_dir, exist_ok=True)
output_filename = os.path.join(output_dir, 'train_log.txt')
logging.basicConfig(level=logging.INFO,
                    filename=output_filename,
                    filemode='w' if not args.resume else 'a')

# set device and random seed
set_device(cfg)
set_random_seed(cfg)
cudnn.benchmark = True

# set classifier
net = get_classifier(cfg, cfg.classifier)
net = net.cuda()

# set optimizers
optimizer = load_optimizer(cfg.optimizer, params=net.parameters())

if cfg.scheduler.type == 'cyclic':
    lr_schedule = lambda t: np.interp([t], cfg.scheduler.args.lr_epochs, cfg.scheduler.args.lr_values)[0]
else:
    lr_schedule = None

start_epoch = 0
if args.resume:
    print("=> loading checkpoint '{}'".format(args.resume))
    ckpt = torch.load(args.resume)
    net.load_state_dict(ckpt['state_dict'])
    optimizer.load_state_dict(ckpt['optimizer'])
    start_epoch = ckpt['epoch']

criterion = torch.nn.CrossEntropyLoss().cuda()
net = move_to_device(net, cfg)

# set stylegan
gan_path = './stylegan_old/pretrain/stylegan_imagenet.pth'
gan = StyleGANGeneratorModel()
state_dict = torch.load(gan_path)
var_name = 'truncation.truncation'
state_dict[var_name] = gan.state_dict()[var_name]
gan.load_state_dict(state_dict)
gan = gan.synthesis
for p in gan.parameters():
    p.requires_grad_(False)
gan = move_to_device(gan, cfg)
model = torch.nn.Sequential(gan, net)

image_attacker = get_attack(cfg.image_attack, net)
latent_attacker = get_attack(cfg.latent_attack, model)

test_attacker = PGDAttack(predict=net,
                          eps=cfg.image_attack.args.eps,
                          eps_iter=cfg.image_attack.args.eps_iter,
                          nb_iter=50,
                          clip_min=-1.0,
                          clip_max=1.0)

# set dataset, dataloader
dataset = get_dataset(cfg)
transform = get_transform(cfg)
trainset = dataset(root=trainset_cfg.path, train=True)
testset = test_dataset(root=testset_cfg.path, train=False)

train_sampler = None
test_sampler = None
if cfg.distributed:
    train_sampler = DistributedSampler(trainset)
    test_sampler = DistributedSampler(testset)

trainloader = DataLoader(trainset,
                         batch_size=trainset_cfg.batch_size,
                         num_workers=4,
                         shuffle=(train_sampler is None),
                         sampler=train_sampler)
testloader = DataLoader(testset,
                        batch_size=testset_cfg.batch_size,
                        num_workers=4,
                        shuffle=False,
                        sampler=test_sampler)


def train(epoch):
    progress_bar = tqdm(trainloader)

    net.train()
    gan.eval()

    image_loss_meter = AverageMeter()
    latent_loss_meter = AverageMeter()
    total_loss_meter = AverageMeter()

    for batch_idx, (images, latents, labels) in enumerate(progress_bar):
        images, latents, labels = images.cuda(), latents.cuda(), labels.cuda()

        lr = cfg.optimizer.args.lr
        if lr_schedule is not None:
            lr = lr_schedule(epoch + (batch_idx + 1) / len(trainloader))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        # with ctx_noparamgrad_and_eval(model):
        #     images_adv = image_attacker.perturb(images, labels)
            # latents_adv = latent_attacker.perturb(latents, labels)

        # images_ladv = gan(latents_adv).detach()

        optimizer.zero_grad()
        total_loss = criterion(net(images), labels)
        # latent_loss = criterion(net(images_ladv), labels)
        # total_loss = 0.5 * image_loss + 0.5 * latent_loss

        total_loss.backward()
        optimizer.step()

        # image_loss_meter.update(image_loss.item(), images.size(0))
        # latent_loss_meter.update(latent_loss.item(), images.size(0))
        total_loss_meter.update(total_loss.item(), images.size(0))

        progress_bar.set_description(
            'Train Epoch: [{0}] | '
            'Total Loss: {t_loss.val:.3f} | '
            'lr: {1:.6f}'.format(
                epoch,
                lr,
                t_loss=total_loss_meter))


def test(epoch):
    progress_bar = tqdm(testloader)
    net.eval()

    acc_clean = AverageMeter()
    acc_adv = AverageMeter()

    for batch_idx, (images, labels) in enumerate(progress_bar):
        images, labels = images.cuda(), labels.cuda()
        with ctx_noparamgrad_and_eval(net):
            images_adv = test_attacker.perturb(images, labels)

            pred_clean = net(images).argmax(dim=1)
            pred_adv = net(images_adv).argmax(dim=1)

        acc_clean.update((pred_clean == labels).float().mean().item() * 100.0, images.size(0))
        acc_adv.update((pred_adv == labels).float().mean().item() * 100.0, images.size(0))

        progress_bar.set_description(
            'Test Epoch: [{0}] '
            'Clean Acc: {acc_clean.val:.3f} ({acc_clean.avg:.3f}) '
            'Adv Acc: {acc_adv.val:.3f} ({acc_adv.avg:.3f}) '.format(epoch, acc_clean=acc_clean, acc_adv=acc_adv))

    logging.info(f'Epoch: {epoch} | Clean: {acc_clean.avg:.2f} % | Adv: {acc_adv.avg:.2f} %')


for epoch in range(start_epoch, cfg.num_epochs):
    if cfg.distributed:
        train_sampler.set_epoch(epoch)
    train(epoch)

    if (epoch + 1) % 2 == 0:
        test(epoch)

    checkpoint_path = os.path.join(output_dir, f'classifier-{epoch:03d}.pt')
    torch.save({
        'epoch': epoch + 1,
        'state_dict': net.module.state_dict(),
        'optimizer': optimizer.state_dict()
    }, checkpoint_path)

    shutil.copyfile(checkpoint_path, os.path.join(output_dir, 'classifier.pt'))



