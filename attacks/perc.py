from typing import Tuple, Optional
import torch
import torch.nn as nn
from math import pi, cos
from advertorch.utils import to_one_hot

from attacks.utils import rgb2lab_diff, ciede2000_diff


TARGET_MULT = 10000.0


class PerC(object):
    """
    PerC_AL: Alternating Loss of Classification and Color Differences to achieve imperceptibile perturbations with few iterations.
    Parameters
    ----------
    max_iterations : int
        Number of iterations for the optimization.
    alpha_l_init: float
        step size for updating perturbations with respect to classification loss
    alpha_c_init: float
        step size for updating perturbations with respect to perceptual color differences
    confidence : float, optional
        Confidence of the adversary for Carlini's loss, in term of distance between logits.
        Note that this approach only supports confidence setting in an untargeted case
    device : torch.device, optional
        Device on which to perform the adversary.
    """

    def __init__(self,
                 predict,
                 max_iterations: int=100,
                 alpha_l_init: float=1.,
                 alpha_c_init: float=0.1,  # or 0.5
                 confidence: float=0,
                 targeted: bool=False
                 ) -> None:
        self.predict = predict
        self.max_iterations = max_iterations
        self.alpha_l_init = alpha_l_init
        self.alpha_c_init = alpha_c_init
        self.confidence = confidence
        self.targeted = targeted
        self.device = torch.device('cuda')

    def perturb(self, inputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Performs the adversary of the model given the inputs and labels.
        Parameters
        """

        # scale to [0, 1]
        inputs = (inputs.clamp(-1, 1) * 0.5 + 0.5).clamp(0, 1).detach().clone()
        X_adv_round_best = inputs.clone()

        alpha_l_min = self.alpha_l_init / 100
        alpha_c_min = self.alpha_c_init / 10
        multiplier = -1 if self.targeted else 1

        inputs_LAB = rgb2lab_diff(inputs, self.device)
        batch_size = inputs.shape[0]
        delta = torch.zeros_like(inputs, requires_grad=True)
        mask_isadv = torch.zeros(batch_size, dtype=torch.bool).to(self.device)
        color_l2_delta_bound_best = (torch.ones(batch_size)*100000).to(self.device)

        if not self.targeted and self.confidence != 0:
            # TODO: have num_classes as class member. However, it is not compatible with other attacks in Advertorch
            labels_onehot = to_one_hot(labels, 10)
        if self.targeted and self.confidence != 0:
            print('Only support setting confidence in untargeted case!')
            return
        for i in range(self.max_iterations):
            # cosine annealing for alpha_l_init and alpha_c_init
            alpha_c = alpha_c_min + 0.5*(self.alpha_c_init-alpha_c_min)*(1+cos(i/self.max_iterations*pi))
            alpha_l = alpha_l_min + 0.5*(self.alpha_l_init-alpha_l_min)*(1+cos(i/self.max_iterations*pi))
            loss = multiplier * nn.CrossEntropyLoss(reduction='sum')(self.predict((inputs + delta) * 2. - 1.), labels)
            loss.backward()
            grad_a = delta.grad.clone()
            delta.grad.zero_()
            delta.data[~mask_isadv] = delta.data[~mask_isadv]+alpha_l*(grad_a.permute(1,2,3,0)/torch.norm(grad_a.view(batch_size,-1),dim=1)).permute(3,0,1,2)[~mask_isadv]
            d_map = ciede2000_diff(inputs_LAB, rgb2lab_diff(inputs+delta, self.device), self.device).unsqueeze(1)
            color_dis = torch.norm(d_map.view(batch_size, -1), dim=1)
            color_loss = color_dis.sum()
            color_loss.backward()
            grad_color = delta.grad.clone()
            delta.grad.zero_()
            delta.data[mask_isadv]=delta.data[mask_isadv]-alpha_c* (grad_color.permute(1,2,3,0)/torch.norm(grad_color.view(batch_size,-1),dim=1)).permute(3,0,1,2)[mask_isadv]

            delta.data = (inputs + delta.data).clamp(0, 1) - inputs
            X_adv_round = inputs + delta.data

            if not self.targeted and self.confidence != 0:
                logits = self.predict(X_adv_round * 2. - 1.)
                real = (labels_onehot * logits).sum(dim=1)

                # TODO: make loss modular, write a loss class
                other = ((1.0 - labels_onehot) * logits - (labels_onehot * TARGET_MULT)).max(1)[0]
                mask_isadv = (real - other) <= -40
            elif self.confidence == 0:
                if self.targeted:
                    mask_isadv = torch.argmax(self.predict(X_adv_round * 2. - 1.), dim=1) == labels
                else:
                    mask_isadv = torch.argmax(self.predict(X_adv_round * 2. - 1.), dim=1) != labels
            mask_best = (color_dis.data < color_l2_delta_bound_best)
            mask = mask_best * mask_isadv
            color_l2_delta_bound_best[mask] = color_dis.data[mask]
            X_adv_round_best[mask] = X_adv_round[mask]

        return X_adv_round_best * 2. - 1.
