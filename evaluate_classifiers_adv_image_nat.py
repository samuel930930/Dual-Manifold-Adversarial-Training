import os
import logging
from tqdm import tqdm
import argparse
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from utils.loaders import \
    set_device, move_to_device, set_random_seed, load_config, get_optimizer, \
    get_scheduler, get_classifier, get_dataset, get_transform, get_attack
from utils.metrics import AverageMeter
from datasets.mixed10_natural import Dataset as test_dataset

# parse command line options
parser = argparse.ArgumentParser(description="Evaluate classifiers on image-domain attacks")
parser.add_argument("--config", default="experiments/classifiers/fimagenet_style_adv_pgd5_pgd5-2_nat.yml")
parser.add_argument("--eval", default="experiments/evaluations/pgd_image_4px.yml")
parser.add_argument("--epoch", default="")
parser.add_argument("--vis", action="store_true")
args = parser.parse_args()

cfg = load_config(args.config)
eval_cfg = load_config(args.eval)

testset_cfg = cfg.dataset.test
print(cfg)

output_dir = os.path.join(cfg.classifier.path, eval_cfg.output_path)
os.makedirs(output_dir, exist_ok=True)
if args.epoch:
    checkpoint_path = os.path.join(cfg.classifier.path, f'classifier-{int(args.epoch):03d}.pt')
    output_filename = os.path.join(output_dir, f'metrics-{int(args.epoch):03d}.txt')
else:
    checkpoint_path = os.path.join(cfg.classifier.path, 'classifier.pt')
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

# set dataset, dataloader
dataset = get_dataset(cfg)
transform = get_transform(cfg)
testset = test_dataset(root=testset_cfg.path, train=False)
testloader = DataLoader(testset,
                        batch_size=16,
                        num_workers=4,
                        shuffle=False)

attacker = get_attack(eval_cfg.attack, net)

acc_clean = AverageMeter()
acc_adv_image = AverageMeter()

progress_bar = tqdm(testloader)

for i, (images, labels) in enumerate(progress_bar):
    images, labels = images.cuda(), labels.cuda()

    images_adv = attacker.perturb(images, labels)

    with torch.no_grad():
        pred_clean = net(images).argmax(dim=1)
        pred_adv = net(images_adv).argmax(dim=1)

        acc_clean.update((pred_clean == labels).float().mean().item() * 100.0, images.size(0))
        acc_adv_image.update((pred_adv == labels).float().mean().item() * 100.0, images.size(0))

    # image_filename = os.path.join(output_dir, f'batch_{i:03d}.pt')
    # torch.save({
    #     'images': images.cpu(),
    #     'images_adv': images_adv.cpu(),
    #     'labels': labels.cpu()
    # }, image_filename)

    progress_bar.set_description(
        '{0} | '
        'Clean: {acc_clean.avg:.2f} % | '
        'Adv: {acc_adv.avg:.2f} %'.format(
            eval_cfg.output_path,
            acc_clean=acc_clean,
            acc_adv=acc_adv_image)
    )
logging.info(f'Clean: {acc_clean.avg:.2f} %')
logging.info(f'Adv: {acc_adv_image.avg:.2f} %')



