import os
import sys
import logging
import torch
import torchvision.transforms as transforms

import models
from . import sgmcmc_optim
from . import datasets


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


def get_dataset(dataset):
    ''' below are cv datasets '''
    if dataset == 'mnist' or dataset == 'fashion-mnist':
        normalize = transforms.Normalize((0.5,), (1.,))
        transform_train = transforms.Compose(
            [transforms.Resize(32), transforms.ToTensor(), normalize])
        transform_test = transforms.Compose(
            [transforms.Resize(32), transforms.ToTensor(), normalize])
    else:
        normalize = transforms.Normalize((0.5,0.5,0.5,), (1.,1.,1.,))
        transform_train = transforms.Compose([
            transforms.ToTensor(), normalize])
        transform_test = transforms.Compose(
            [transforms.ToTensor(), normalize])

    if dataset == 'mnist':
        trainset = datasets.MNIST('./data', True, transform_train)
        testset = datasets.MNIST('./data', False, transform_test)
    elif dataset == 'fashion-mnist':
        trainset = datasets.FashionMNIST('./data', True, transform_train)
        testset = datasets.FashionMNIST('./data', False, transform_test)
    elif dataset == 'cifar10':
        trainset = datasets.CIFAR10('./data', True, transform_train)
        testset = datasets.CIFAR10('./data', False, transform_test)
    elif dataset == 'cifar100':
        trainset = datasets.CIFAR100('./data', True, transform_train)
        testset = datasets.CIFAR100('./data', False, transform_test)
    else:
        raise ValueError('dataset {} is not supported'.format(dataset))

    return [trainset, testset]


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


def get_mcmc_bnn_arch(arch, dataset, prior_sig):
    if dataset == 'mnist' or dataset == 'fashion-mnist':
        in_dims, out_dims = 1, 10
    elif dataset == 'cifar10':
        in_dims, out_dims = 3, 10
    elif dataset == 'cifar100':
        in_dims, out_dims = 3, 100
    else:
        raise ValueError('dataset {} is not supported'.format(dataset))

    if arch == 'mlp':
        return models.mcmcMLP(prior_sig, in_dims)
    elif arch == 'lenet':
        return models.mcmcLeNet(prior_sig, in_dims)
    elif arch == 'small-cnn':
        return models.mcmcSmallCNN(prior_sig, in_dims, out_dims)

    raise ValueError('arch {} is not supported'.format(arch))


def load_pre_mcmc_bnn(model, arch, dataset, state_dict):
    model_state_dict = model.state_dict()
    new_state_dict = dict()

    for name, param in state_dict.items():
        if dataset == 'cifar10':
            if (arch=='small-cnn') and ('fc2' in name): continue
        new_state_dict[name] = param

    for name, param in new_state_dict.items():
        model_state_dict[name].copy_(param)
