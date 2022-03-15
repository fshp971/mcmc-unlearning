import numpy as np
import pickle
import os
import sys
import logging
import torch
import torchvision.transforms as transforms

import models
from . import sgmcmc_optim
from .data import Dataset


class AverageMeter():
    def __init__(self):
        self.cnt = 0
        self.sum = 0
        self.mean = 0

    def update(self, val, cnt):
        self.cnt += cnt
        self.sum += val * cnt
        self.mean = self.sum / self.cnt

    def average(self):
        return self.mean
    
    def total(self):
        return self.sum


def add_log(log, key, value):
    if key not in log.keys():
        log[key] = []
    log[key].append(value)


def get_GMM2d(path='./data'):
    with open('{}/GMMs/gmm-2d-syn-set.pkl'.format(path), 'rb') as f:
        raw_dataset = pickle.load(f)

    return Dataset(raw_dataset['x'], raw_dataset['y'])


''' we use a fixed removal set for the GMM unlearning experiments '''
def get_forget_idx(dataset, kill_num):
    ''' for special case '''
    randidx = []
    hf = kill_num // 2

    idx = np.where(dataset.y==0)[0]
    sorted_idx = dataset.x[:,0][idx].argsort()
    randidx.append( idx[sorted_idx[len(sorted_idx)-hf:]] )

    idx = np.where(dataset.y==2)[0]
    sorted_idx = dataset.x[:,1][idx].argsort()
    # randidx.append( idx[sorted_idx[len(sorted_idx)-(kill_num-hf):]] )
    randidx.append( idx[sorted_idx[:(kill_num-hf)]] )

    randidx = np.concatenate(randidx)
    randidx = np.random.permutation(randidx)
    return randidx


def get_optim(parameters, opt, **kwargs):
    lr = kwargs['lr']
    weight_decay = kwargs['weight_decay']

    if opt == 'sgd':
        momentum = kwargs['momentum']
        return torch.optim.SGD(parameters, momentum=momentum,
                    lr=lr, weight_decay=weight_decay)

    elif opt == 'adam':
        return torch.optim.Adam(parameters,
                    lr=lr, weight_decay=weight_decay)

    elif opt == 'sgld':
        return sgmcmc_optim.SGLD(parameters, lr=lr)

    elif opt == 'sghmc':
        alpha = kwargs['sghmc_alpha']
        return sgmcmc_optim.SGHMC(parameters, lr=lr, alpha=alpha)

    raise ValueError('optim method {} is not supported'.format(opt))


def adjust_learning_rate(optimizer, epoch, lr, lr_decay_rate, lr_decay_freq):
    if (lr_decay_rate is None) or (lr_decay_freq is None):
        return

    lr = lr * (lr_decay_rate ** (epoch // lr_decay_freq))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def generic_init(args):
    if os.path.exists(args.save_dir) == False:
        os.makedirs(args.save_dir)

    fmt = '%(asctime)s %(name)s:%(levelname)s:  %(message)s'
    formatter = logging.Formatter(
        fmt, datefmt='%Y-%m-%d %H:%M:%S')

    fh = logging.FileHandler(
        '{}/{}_log.txt'.format(args.save_dir, args.save_name), mode='w')
    fh.setFormatter(formatter)

    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=fmt, datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger()
    logger.addHandler(fh)

    logger.info('Arguments')
    for arg in vars(args):
        logger.info('    {:<22}        {}'.format(arg+':', getattr(args,arg)) )
    logger.info('')

    return logger


def evaluate(model, loader, cpu):
    loss = AverageMeter()
    ''' log_pp stands for log predictive probability '''
    log_pp = AverageMeter()

    model.eval()
    for x, _ in loader:
        if not cpu: x = x.cuda()

        with torch.no_grad():
            _y = model(x)
            lo = model.loss(x, _y)

        loss.update(lo.item(), len(x))
        log_pp.update(_y.max(dim=1)[0].data.log().mean().item(), len(x))

    return loss.average(), log_pp.average()


def save_checkpoint(save_dir, save_name, log, model, optimizer, save_cluster=True):
    with open('{}/{}-log.pkl'.format(save_dir, save_name), 'wb') as f:
        pickle.dump(log, f)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        }, '{}/{}-model.pkl'.format(save_dir, save_name))

    if save_cluster:
        with open('{}/{}-cluster.pkl'.format(save_dir, save_name), 'wb') as f:
            pickle.dump(model.dump_numpy_dict(), f)
