import os
import argparse
from tqdm import tqdm
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from utils.loaders import \
    set_random_seed, load_config, get_defense, \
    get_transform, get_dataset, get_classifier
from utils.losses import per_pixel_l2_dist
from torchvision.utils import save_image
from advertorch.attacks import FGSM

# parse command line options
parser = argparse.ArgumentParser(description="Whitebox FGSM")
parser.add_argument("--config", default="experiments/defenses/mnist/defensegan.yml", help="config file")
parser.add_argument("--eps", type=int, default=8)
parser.add_argument("--world_size", type=int, default=1)
parser.add_argument("--rank", type=int, default=0)
args = parser.parse_args()

cfg = load_config(args.config)
dataset_cfg = cfg.dataset.test
print(cfg)

# set experiment paths
output_dir = (cfg.defense.path).strip('/') + f'_fgsm_{args.eps}'
vis_dir = os.path.join(output_dir, 'vis')
result_dir = os.path.join(output_dir, 'result')
os.makedirs(vis_dir, exist_ok=True)
os.makedirs(result_dir, exist_ok=True)

# set random seed
set_random_seed(cfg)
cudnn.benchmark = True

# load defense
proj_fn = get_defense(cfg, cfg.defense).cuda()
proj_fn.eval()

# load dataset
dataset = get_dataset(cfg)
transform = get_transform(cfg)
testset = dataset(root=dataset_cfg.path, train=False, transform=transform.default)
testloader = DataLoader(testset, batch_size=dataset_cfg.batch_size, num_workers=0, shuffle=False)
progress_bar = tqdm(testloader)
n_batch = len(testloader)
chunk_size = n_batch // args.world_size
start_ind = args.rank * chunk_size
end_ind = (args.rank + 1) * chunk_size

# load classifier
predict = get_classifier(cfg, cfg.classifier)
state_dict = torch.load(cfg.classifier.ckpt)
predict.load_state_dict(state_dict)
for p in predict.parameters():
    p.requires_grad_(False)
predict = torch.nn.Sequential(transform.classifier_preprocess_layer, predict).cuda()
predict.eval()

# create attacker
attacker = FGSM(predict=predict, eps=args.eps/255.0)

total = 0
correct_clean = 0
correct_adv = 0
correct_def = 0
for i, (images, labels) in enumerate(progress_bar):
    if i < start_ind or i >= end_ind:
        continue

    images, labels = images.cuda(), labels.cuda()
    result_path = os.path.join(result_dir, 'batch_{:04d}.pt'.format(i))
    if os.path.isfile(result_path):
        result_dict = torch.load(result_path)
        images_adv = result_dict['input'].cuda()
        images_def = result_dict['rec'].cuda()

    else:
        images_adv = attacker.perturb(images)
        images_def, z_def = proj_fn(images_adv)
        torch.save({'input': images_adv,
                    'rec': images_def,
                    'z_rec': z_def}, result_path)

    l2_dist = per_pixel_l2_dist(images, images_adv)

    if i % 1 == 0:
        clean_path = os.path.join(vis_dir, 'batch_{:04d}_clean.png'.format(i))
        adv_path = os.path.join(vis_dir, 'batch_{:04d}_adv.png'.format(i))
        def_path = os.path.join(vis_dir, 'batch_{:04d}_def.png'.format(i))
        save_image(images, clean_path, nrow=10, padding=2)
        save_image(images_adv, adv_path, nrow=10, padding=2)
        save_image(images_def, def_path, nrow=10, padding=2)

    with torch.no_grad():
        pred_clean = predict(images).argmax(dim=1)
        pred_adv = predict(images_adv).argmax(dim=1)
        pred_def = predict(images_def).argmax(dim=1)

        total += labels.size(0)
        correct_clean += (pred_clean == labels).float().sum().item()
        correct_adv += (pred_adv == labels).float().sum().item()
        correct_def += (pred_def == labels).float().sum().item()

    progress_bar.set_description('Clean Acc: {:.3f} | '
                                 'Adv Acc: {:.3f} | '
                                 'Def Acc: {:.3f} | '
                                 'L2 dist: {:.6f}'.format(100. * correct_clean / total,
                                                          100. * correct_adv / total,
                                                          100. * correct_def / total,
                                                          l2_dist))

# save numerical results
if args.world_size > 1:
    acc_fn = f'acc_{args.rank}_{args.world_size}.txt'
else:
    acc_fn = 'acc.txt'
with open(os.path.join(output_dir, acc_fn), 'w') as f:
    out = 'Clean Acc: {:.3f} | ' \
          'Adv Acc: {:.3f} | ' \
          'Def Acc: {:.3f}'.format(100. * correct_clean / total,
                                   100. * correct_adv / total,
                                   100. * correct_def / total)
    f.writelines(out)
