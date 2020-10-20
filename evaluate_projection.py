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

# parse command line options
parser = argparse.ArgumentParser(description="Evaluate Projected Samples")
parser.add_argument("--config", default="experiments/defenses/mnist/defensegan.yml", help="config file")
parser.add_argument("--world_size", type=int, default=1)
parser.add_argument("--rank", type=int, default=0)
args = parser.parse_args()

cfg = load_config(args.config)
dataset_cfg = cfg.dataset.test
print(cfg)

# set experiment paths
output_dir = (cfg.defense.path).strip('/') + '_clean'
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
predict = torch.nn.Sequential(transform.classifier_preprocess_layer, predict).cuda()
predict.eval()

total = 0
correct_clean = 0
correct_rec = 0

for i, (images, labels) in enumerate(progress_bar):
    if i < start_ind or i >= end_ind:
        continue

    images, labels = images.cuda(), labels.cuda()
    result_path = os.path.join(result_dir, 'batch_{:04d}.pt'.format(i))
    if os.path.isfile(result_path):
        result_dict = torch.load(result_path)
        images_rec = result_dict['rec'].cuda()

    else:
        images_rec, z_rec = proj_fn(images)
        torch.save({'input': images,
                    'rec': images_rec,
                    'z_rec': z_rec}, result_path)

    # visualization
    clean_vis_path = os.path.join(vis_dir, 'batch_{:04d}_clean.png'.format(i))
    rec_vis_path = os.path.join(vis_dir, 'batch_{:04d}_rec.png'.format(i))
    save_image(images, clean_vis_path, nrow=10, padding=2)
    save_image(images_rec, rec_vis_path, nrow=10, padding=2)

    with torch.no_grad():
        logits_clean = predict(images)
        logits_rec = predict(images_rec)
        pred_clean = logits_clean.argmax(dim=1)
        pred_rec = logits_rec.argmax(dim=1)

        total += labels.size(0)
        correct_clean += (pred_clean == labels).float().sum().item()
        correct_rec += (pred_rec == labels).float().sum().item()

    progress_bar.set_description('Clean Acc: {:.3f} | '
                                 'Rec Acc: {:.3f}'.format(100. * correct_clean / total,
                                                          100. * correct_rec / total))

# save numerical results
if args.world_size > 1:
    acc_fn = f'acc_{args.rank}_{args.world_size}.txt'
else:
    acc_fn = 'acc.txt'
with open(os.path.join(output_dir, acc_fn), 'w') as f:
    out = 'Clean Acc: {:.3f} | ' \
          'Rec Acc: {:.3f}'.format(100. * correct_clean / total,
                                   100. * correct_rec / total)
    f.writelines(out)
