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

    parser.add_argument('--rm-idx-path', type=str, default=None)
    parser.add_argument('--finetune-steps', type=int, default=1)

    return parser.parse_args()


def get_forget_idx(dataset, kill_num):
    kill_val = 0

    if 'targets' in vars(dataset).keys():
        labels = np.array(dataset.targets)
    elif 'labels' in vars(dataset).keys():
        labels = np.array(dataset.labels)
    else:
        raise NotImplementedError

    randidx = np.random.permutation( np.where(labels==kill_val)[0] )
    return randidx[:kill_num]


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
            lo = - model.log_prior() + F.cross_entropy(_y,y) * n
            lo = lo.item()
            ac = (_y.argmax(dim=1) == y).sum().item() / len(y)

        loss.update(lo, len(y))
        acc.update(ac, len(y))

    return loss.average(), acc.average()


def forget_eval_one_time(model, train_loader, forgetted_train_loader, test_loader, log):
    remain_train_loss, remain_train_acc = evaluate(model, train_loader, args.cpu)
    forgetted_train_loss, forgetted_train_acc = evaluate(model, forgetted_train_loader, args.cpu)
    test_loss, test_acc = evaluate(model, test_loader, args.cpu)

    utils.add_log(log, 'remain_train_loss', remain_train_loss)
    utils.add_log(log, 'remain_train_acc', remain_train_acc)
    utils.add_log(log,'forgetted_train_loss', forgetted_train_loss)
    utils.add_log(log,'forgetted_train_acc', forgetted_train_acc)
    utils.add_log(log, 'test_loss', test_loss)
    utils.add_log(log, 'test_acc', test_acc)

    logger.info('remaining train loss {:.2e} \t train acc {:.2%}'
                .format(remain_train_loss, remain_train_acc))
    logger.info('forgetted train loss {:.2e} \t train acc {:.2%}'
                .format(forgetted_train_loss, forgetted_train_acc))
    logger.info('test loss {:.2e} \t test acc {:.2%}'
                .format(test_loss, test_acc))
    logger.info('')


def save_checkpoint(save_dir, save_name, log, model, optimizer):
    with open('{}/{}-log.pkl'.format(save_dir, save_name), 'wb') as f:
        pickle.dump(log, f)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        }, '{}/{}-model.pkl'.format(save_dir, save_name))


def main(args, logger):
    ''' retrieve lots of data '''
    trainset, testset = utils.get_dataset(args.dataset)

    if args.rm_idx_path is not None:
        with open(args.rm_idx_path, 'rb') as f:
            forgetted_idx = pickle.load(f)
    else:
        forgetted_idx = get_forget_idx(trainset, args.ifs_kill_num)

    forgetted_idx_loader = utils.IndexBatchSampler(
        batch_size=args.ifs_rm_bs, indices=forgetted_idx)

    train_sampler = utils.DataSampler(trainset, args.batch_size)

    train_loader = utils.DataLoader(trainset, args.batch_size)
    train_loader.remove(forgetted_idx)

    forgetted_train_loader = utils.DataLoader(trainset, args.batch_size)
    forgetted_train_loader.set_sampler_indices(forgetted_idx)

    test_loader = utils.DataLoader(testset, args.batch_size)
    ''' end of retrieving data '''

    model = utils.get_mcmc_bnn_arch(args.arch, args.dataset, args.prior_sig)

    if not args.cpu:
        model.cuda()

    args.lr /= len(trainset)
    optimizer = utils.get_optim(model.parameters(), args.optim,
        lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, sghmc_alpha=args.sghmc_alpha)

    model.n = len(train_sampler)

    ''' restore model / sampler '''
    state_dict = torch.load(args.resume_path)
    model.load_state_dict(state_dict['model_state_dict'])
    optimizer.load_state_dict(state_dict['optimizer_state_dict'])

    ''' for backward compatibility '''
    for group in optimizer.param_groups:
        if 'lr_decay' in group:
            group['lr'] *= group['lr_decay']
            group.pop('lr_decay')

    del state_dict

    log = dict()
    log['user_time'] = 0
    utils.add_log(log, 'forgetted_idx', forgetted_idx)

    forget_eval_one_time(model, train_loader, forgetted_train_loader, test_loader, log)

    for ii in forgetted_idx_loader:
        ''' update the number of remaining datums before fine-tuning '''
        train_sampler.remove(ii)
        model.n = len(train_sampler)

        ''' update mcmc sampler '''
        for group in optimizer.param_groups:
            group['lr'] *= (len(train_sampler) + len(ii)) / len(train_sampler)

        ''' start calculation of time '''
        start_time = datetime.now()

        for step in range(args.finetune_steps):
            x, y = next(train_sampler)
            if not args.cpu: x, y = x.cuda(), y.cuda()
            model.train()
            lo = -model.log_prior() + F.cross_entropy(model(x), y) * model.n
            optimizer.zero_grad()
            lo.backward()
            optimizer.step()

        torch.cuda.synchronize()
        end_time = datetime.now()
        user_time = (end_time - start_time).total_seconds()
        ''' end calculation of time '''

        log['user_time'] += user_time

        logger.info('remaining trainset size {}'.format(len(train_sampler)))
        logger.info('user time {:.3f} sec \t'
                    'cumulated user time {:.3f} mins'
                    .format(user_time, log['user_time']/60) )

        forget_eval_one_time(model, train_loader, forgetted_train_loader, test_loader, log)

    save_checkpoint(args.save_dir, args.save_name, log, model, optimizer)

    return


if __name__ == '__main__':
    args = get_args()
    logger = utils.generic_init(args)

    try:
        main(args, logger)
    except Exception as e:
        logger.exception('Unexpected exception! %s', e)
