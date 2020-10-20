import os
import time
import torch
from torchvision.utils import save_image
from model import PixelCNN
from utils import *


ckpt_path = './tf_model/pcnn_889.pth'
ckpt = torch.load(ckpt_path)

model = PixelCNN(nr_resnet=5, nr_filters=160,
                 input_channels=3, nr_logistic_mix=10)
model = torch.nn.DataParallel(model)
model.cuda()

model.load_state_dict(ckpt)
model = model.module
model.eval()

sample_op = lambda x: sample_from_discretized_mix_logistic(x, 10)

with torch.no_grad():
    for n in range(3):
        data = torch.zeros(25, 3, 32, 32)
        data = data.cuda()
        start_time = time.time()
        for i in range(32):
            for j in range(32):
                out = model(data, sample=True)
                out_sample = sample_op(out)
                data[:, :, i, j] = out_sample.data[:, :, i, j]
                print(j)

        print(f'{time.time() - start_time} secs')
        save_image(data, f'./sample_{n}.png', normalize=True, range=(-1., 1.))

