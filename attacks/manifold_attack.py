from advertorch.utils import normalize_by_pnorm, batch_multiply
from advertorch.utils import to_one_hot
import torch
import torch.nn as nn
import torch.nn.functional as F


class ManifoldAttack(object):
    def __init__(self, predict, gan, num_classes, loss_fn=None, eps_iter=0.0005, nb_iter=5, ord='inf'):
        self.predict = predict
        self.gan = gan
        self.num_classes = num_classes
        self.nb_iter = nb_iter
        self.eps_iter = eps_iter
        self.loss_fn = loss_fn
        self.ord = ord
        if self.loss_fn is None:
            self.loss_fn = nn.CrossEntropyLoss(reduction='sum')

    def perturb(self, z, y):
        z = z.detach().clone()
        y = y.detach().clone()

        delta = torch.zeros_like(z)
        delta.requires_grad_()

        for ii in range(self.nb_iter):
            outputs = self.predict(self.gan(z + delta))
            adv_loss = self.loss_fn(outputs, y)
            reg_loss = F.relu(delta.pow(2) - 0.01).mean(dim=-1).sum()
            loss = adv_loss - 100 * reg_loss
            loss.backward()

            if self.ord == 'inf':
                grad_sign = delta.grad.data.sign()
                delta.data = delta.data + batch_multiply(self.eps_iter, grad_sign)
            elif self.ord == 'l2':
                grad = delta.grad.data
                grad = normalize_by_pnorm(grad)
                delta.data = delta.data + batch_multiply(self.eps_iter, grad)
            else:
                delta.data = delta.data + batch_multiply(self.eps_iter, delta.grad.data)

            delta.grad.data.zero_()
            print('[{}/{}] Loss: {:4f}'.format(ii, self.nb_iter, adv_loss.item()))

        return self.gan(z + delta)


