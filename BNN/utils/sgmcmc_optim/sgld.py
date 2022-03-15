import numpy as np
import torch

''' References:
    [1] https://www.ics.uci.edu/~welling/publications/papers/stoclangevin_v6.pdf
    [2] https://github.com/pytorch/pytorch/blob/master/torch/optim/sgd.py
'''
class SGLD(torch.optim.Optimizer):
    def __init__(self, params, lr):
        defaults = dict(lr=lr)
        super(SGLD, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']

            for p in group['params']:
                if p.grad is None:
                    continue
                noise = p.data.clone().normal_(0, np.sqrt(2*lr) )

                p.add_(p.grad, alpha= -lr)
                p.add_(noise)

        return loss

    def set_learning_rate(self, lr):
        for group in self.param_groups:
            group['lr'] = lr
