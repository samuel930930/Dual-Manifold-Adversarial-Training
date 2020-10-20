import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from defenses.pixelcnn.model import PixelCNN
from defenses.pixelcnn.utils import log_sum_exp, log_prob_from_logits
import time
from utils.loaders import get_optimizer, get_scheduler, get_generator, get_transform
from utils.losses import per_pixel_l2_dist


def mix_logistic_to_logits(x, l):
    """
    Bin the output to 256 discrete values. This discretizes the continuous mixture of logistic distribution
    output to categorical distributions.
    """
    device = x.device

    xs = list(x.size())
    ls = list(l.size())
    nr_mix = int(ls[-1] / 10)

    logit_probs = l[:, :, :, :nr_mix]
    l = l[:, :, :, nr_mix:].view(xs + [nr_mix * 3])
    means = l[:, :, :, :, :nr_mix]
    log_scales = torch.clamp(l[:, :, :, :, nr_mix:2 * nr_mix], min=-7.)
    coeffs = F.tanh(l[:, :, :, :, 2 * nr_mix:3 * nr_mix])
    # here and below: getting the means and adjusting them based on preceding
    # sub-pixels
    x = x.view(xs + [1]) + torch.zeros(xs + [nr_mix], requires_grad=False, device=device)
    m2 = (means[:, :, :, 1, :] + coeffs[:, :, :, 0, :]
          * x[:, :, :, 0, :]).view([xs[0], xs[1], xs[2], 1, nr_mix])
    m3 = (means[:, :, :, 2, :] + coeffs[:, :, :, 1, :] * x[:, :, :, 0, :] +
                    coeffs[:, :, :, 2, :] * x[:, :, :, 1, :]).view([xs[0], xs[1], xs[2], 1, nr_mix])
    means = torch.cat((means[:, :, :, 0, :].unsqueeze(3), m2, m3), dim=3)

    inv_stdv = torch.exp(-log_scales)

    colors = torch.linspace(-1., 1., 257, requires_grad=False, device=device)
    color_maps = torch.zeros(xs + [1, 1], requires_grad=False, device=device) + colors
    means = means.unsqueeze(5)
    inv_stdv = inv_stdv.unsqueeze(5)
    color_cdfs = F.sigmoid((color_maps[..., 1:-1] - means) * inv_stdv)
    color_pdfs = color_cdfs[..., 1:] - color_cdfs[..., :-1]
    normalized_0 = (color_maps[..., 1:2] - means) * inv_stdv
    normalized_255 = (color_maps[..., -2:-1] - means) * inv_stdv
    color_log_cdf0 = normalized_0 - F.softplus(normalized_0)
    color_log_cdf255 = -F.softplus(normalized_255)

    color_mids = torch.linspace(-1., 1., 513, requires_grad=False, device=device)[3:-2:2]
    color_mid_maps = torch.zeros(xs + [1, 1], requires_grad=False, device=device) + color_mids
    color_mid_maps = inv_stdv * (color_mid_maps - means)
    color_mid_map_log_pdfs = color_mid_maps - log_scales.unsqueeze(5) - 2. * F.softplus(color_mid_maps)

    color_log_pdfs = torch.where(color_pdfs > 1e-5, torch.log(torch.clamp(color_pdfs, min=1e-12)),
                                 color_mid_map_log_pdfs - np.log(127.5))

    # color_log_pdfs = tf.log(tf.maximum(color_pdfs, 1e-12))

    color_log_probs = torch.cat([color_log_cdf0, color_log_pdfs, color_log_cdf255], dim=5)

    color_log_probs = color_log_probs + log_prob_from_logits(logit_probs)[:, :, :, None, :, None]

    return log_sum_exp(color_log_probs, axis=4)


def decode_from_softmax_with_box_constraint(x, l, eps):
    '''
    Greedy decoding. Equivalent to beam search with size = 1
    Note: The output is a unnormalized image
    '''
    device = x.device
    x = x.permute(0, 2, 3, 1).contiguous()
    l = l.permute(0, 2, 3, 1).contiguous()

    recover_img = (x * 127.5 + 127.5).int()
    lb = torch.clamp(recover_img - eps, min=0)
    ub = torch.clamp(recover_img + eps, max=255)
    template = torch.arange(0, 256, dtype=torch.int32, device=device) + \
               torch.zeros_like(recover_img, dtype=torch.int32)[..., None]

    lb = lb[..., None] + torch.zeros_like(template, dtype=torch.int32)
    ub = ub[..., None] + torch.zeros_like(template, dtype=torch.int32)
    template = ((template < lb) | (template > ub)).float()

    logits0 = mix_logistic_to_logits(x, l)
    logits0 -= template * 1e30
    x0 = (torch.argmax(logits0, dim=4).float() - 127.5) / 127.5

    new_x = torch.stack([x0[:, :, :, 0], x[:, :, :, 1], x[:, :, :, 2]], dim=3)
    logits1 = mix_logistic_to_logits(new_x, l)
    logits1 -= template * 1e30
    x1 = (torch.argmax(logits1, dim=4).float() - 127.5) / 127.5

    new_x = torch.stack([x0[:, :, :, 0], x1[:, :, :, 1], x[:, :, :, 2]], dim=3)
    logits2 = mix_logistic_to_logits(new_x, l)
    logits2 -= template * 1e30
    x2 = (torch.argmax(logits2, dim=4).float() - 127.5) / 127.5

    out = torch.stack([x0[:, :, :, 0], x1[:, :, :, 1], x2[:, :, :, 2]], dim=3)
    out = out.permute(0, 3, 1, 2)
    return out


def carlini_decode():
    pass


class PixelDefendProjection(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, pixelcnn, eps=16):

        purified = x.detach().clone()
        # x from [-1, 1] to [0, 255]: (x + 1) / 2 * 255
        for row in range(32):
            for col in range(32):
                l = pixelcnn(purified, sample=True)
                p = decode_from_softmax_with_box_constraint(purified, l, eps=eps)

                purified[:, row:(row+1), col:(col+1), :] = p[:, row:(row+1), col:(col+1), :]

        return (purified + 1) * 0.5, torch.zeros(1)

    @staticmethod
    def backward(ctx, grad_output, grad_z):
        grad_input = grad_output.clone()
        return grad_input, None, None


class Preprocessor(nn.Module):
    def __init__(self, cfg):
        super(Preprocessor, self).__init__()

        # load pixelcnn
        pixelcnn = PixelCNN(nr_resnet=5, nr_filters=160, input_channels=3, nr_logistic_mix=10)
        pixelcnn = torch.nn.DataParallel(pixelcnn)
        state_dict = torch.load(cfg.pixelcnn.ckpt)
        pixelcnn.load_state_dict(state_dict)
        self.pixelcnn = pixelcnn.module.cuda()
        for p in self.pixelcnn.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        # process x to be in [-1, 1]
        x = x * 2. - 1.

        return PixelDefendProjection.apply(x, self.pixelcnn)
