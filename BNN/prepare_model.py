from datetime import datetime
import os
import pickle
import argparse
import numpy as np
import torch
import torch.nn.functional as F

import utils
import models


def get_args():
    parser = argparse.ArgumentParser()
    utils.add_shared_args(parser)

    parser.add_argument('--save-freq', type=int, default=1000,
                        help='frequency of evaluating during training')

    parser.add_argument('--pretrain-dataset', type=str, default=None,
                        choices=['mnist', 'fashion-mnist', 'cifar10', 'cifar100'])

    parser.add_argument('--finetune-dataset', type=str, default=None,
                        choices=['mnist', 'fashion-mnist', 'cifar10', 'cifar100'])

    return parser.parse_args()


def get_arch(arch, dataset):
    if dataset == 'mnist' or dataset == 'fashion-mnist':
        in_dims, out_dims = 1, 10
    elif dataset == 'cifar10':
        in_dims, out_dims = 3, 10
    elif dataset == 'cifar100':
        in_dims, out_dims = 3, 100
    else:
        raise ValueError('dataset {} is not supported'.format(dataset))

    if arch == 'mlp':
        raise NotImplementedError
        return models.MLP(in_dims)

    elif arch == 'lenet':
        return models.LeNet(in_dims, out_dims)

    elif arch == 'small-cnn':
        return models.SmallCNN(in_dims, out_dims)

    elif arch == 'all-cnn':
        raise NotImplementedError
        return models.AllCNN(in_dims)

    elif arch == 'vgg9':
        raise NotImplementedError
        return models.VGG9(in_dims)

    elif arch == 'resnet18':
        return models.ResNet18(in_dims, out_dims)

    raise ValueError('arch {} is not supported'.format(arch))


def evaluate(model, loader, cpu):
    ''' average log predictive probability '''
    loss = utils.AverageMeter()
    acc = utils.AverageMeter()

    n = len(loader.sampler.indices)

    model.eval()
    for x, y in loader:
        if not cpu: x, y = x.cuda(), y.cuda()

        with torch.no_grad():
            _y = model(x)
            lo = F.cross_entropy(_y, y)
            lo = lo.item()
            ac = (_y.argmax(dim=1) == y).sum().item() / len(y)

        loss.update(lo, len(y))
        acc.update(ac, len(y))

    return loss.average(), acc.average()


def save_checkpoint(save_dir, save_name, log, model, optimizer):
    with open('{}/{}-log.pkl'.format(save_dir, save_name), 'wb') as f:
        pickle.dump(log, f)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        }, '{}/{}-model.pkl'.format(save_dir, save_name))


def main(args, logger):
    ''' retrieve lots of data '''
    trainset, testset = utils.get_dataset(args.pretrain_dataset)

    train_sampler = utils.DataSampler(trainset, args.batch_size)
    train_loader = utils.DataLoader(trainset, args.batch_size)
    test_loader = utils.DataLoader(testset, args.batch_size)
    ''' end of retrieving data '''

    model = get_arch(args.arch, args.pretrain_dataset)

    optimizer = utils.get_optim(model.parameters(), args.optim,
        lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    criterion = torch.nn.CrossEntropyLoss()

    if not args.cpu:
        model.cuda()

    log = dict()
    log['user_time'] = 0

    for step in range(args.burn_in_steps):

        x, y = next(train_sampler)
        if not args.cpu: x, y = x.cuda(), y.cuda()

        start_time = datetime.now()

        ''' non-bayesian nn '''
        model.train()
        _y = model(x)
        lo = criterion(_y, y)
        optimizer.zero_grad()
        lo.backward()
        optimizer.step()
        loss = lo.item()
        acc = (_y.argmax(dim=1) == y).sum().item() / len(y)

        torch.cuda.synchronize()
        end_time = datetime.now()
        user_time = (end_time - start_time).total_seconds()
        log['user_time'] += user_time

        utils.add_log(log, 'burn_in_loss', loss)
        utils.add_log(log, 'burn_in_acc', acc)

        if (step+1) % args.eval_freq == 0:
            logger.info('burn-in step [{}/{}]:'
                        .format(step+1, args.burn_in_steps))
            logger.info('user time {:.3f} sec \t'
                        'cumulated user time {:.3f} mins'
                        .format(user_time, log['user_time']/60) )
            logger.info('burn-in loss {:.2e} \t'
                        'burn-in acc {:.2%}'
                        .format(loss, acc) )

            test_loss, test_acc = evaluate(model, test_loader, args.cpu)
            logger.info('test loss {:.2e} \t'
                        'test acc {:.2%}'
                        .format(test_loss, test_acc) )
            utils.add_log(log, 'test_loss', test_loss)
            utils.add_log(log, 'test_acc', test_acc)

            train_loss, train_acc = evaluate(model, train_loader, args.cpu)
            logger.info('(maybe remain) train loss {:.2e} \t'
                        'train acc {:.2%}'
                        .format(train_loss, train_acc) )
            utils.add_log(log, '(remain) train_loss', train_loss)
            utils.add_log(log, '(remain) train_acc', train_acc)

            logger.info('')

        if (step+1) % args.save_freq == 0:
            save_checkpoint(args.save_dir, '{}-ckpt-{}'.format(args.save_name, step+1), log, model, optimizer)

    fin_model = get_arch(args.arch, args.finetune_dataset)
    utils.load_pre_mcmc_bnn(fin_model, args.arch, args.finetune_dataset, model.state_dict())

    save_checkpoint(args.save_dir, '{}-fin'.format(args.save_name), log, fin_model, optimizer)

    return


if __name__ == '__main__':
    args = get_args()
    logger = utils.generic_init(args)

    try:
        main(args, logger)
    except Exception as e:
        logger.exception('Unexpected exception! %s', e)
