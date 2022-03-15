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
    parser.add_argument('--init-model-path', type=str, default=None)

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


def save_checkpoint(save_dir, save_name, log, model, optimizer):
    with open('{}/{}-log.pkl'.format(save_dir, save_name), 'wb') as f:
        pickle.dump(log, f)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        }, '{}/{}-model.pkl'.format(save_dir, save_name))


def adjust_learning_rate(optimizer, step, init_lr, lr_decay_rate, lr_decay_freq):
    lr = init_lr * ( lr_decay_rate ** (step // lr_decay_freq) )
    for group in optimizer.param_groups:
        group['lr'] = lr


def adjust_step_size(optimizer, step, sp_init, sp_decay_exp):
    sp_rate = step ** sp_decay_exp
    for group in optimizer.param_groups:
        group['lr'] = sp_init * sp_rate


def main(args, logger):
    ''' retrieve lots of data '''
    trainset, testset = utils.get_dataset(args.dataset)

    if args.rm_idx_path is not None:
        with open(args.rm_idx_path, 'rb') as f:
            forgetted_idx = pickle.load(f)
    else:
        forgetted_idx = get_forget_idx(trainset, args.ifs_kill_num)

    train_sampler = utils.DataSampler(trainset, args.batch_size)
    train_sampler.remove(forgetted_idx)

    train_loader = utils.DataLoader(trainset, args.batch_size)
    train_loader.remove(forgetted_idx)

    forgetted_train_loader = utils.DataLoader(trainset, args.batch_size)
    forgetted_train_loader.set_sampler_indices(forgetted_idx)

    test_loader = utils.DataLoader(testset, args.batch_size)
    ''' end of retrieving data '''

    model = utils.get_mcmc_bnn_arch(args.arch, args.dataset, args.prior_sig)

    if args.init_model_path is not None:
        state_dict = torch.load(args.init_model_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict['model_state_dict'])
        del state_dict

    args.lr /= len(train_sampler)
    args.sp_init /= len(train_sampler)
    optim_params = {'lr':args.lr, 'momentum':args.momentum, 'weight_decay':args.weight_decay, 'sghmc_alpha':args.sghmc_alpha}
    optimizer = utils.get_optim(model.parameters(), 'sgd', **optim_params)

    model.n = len(train_sampler)
    if not args.cpu:
        model.cuda()

    log = dict()
    log['user_time'] = 0
    utils.add_log(log, 'forgetted_idx', forgetted_idx)

    sampling_steps = 0
    for step in range(args.burn_in_steps):
        ''' adjust learning rate / step-size '''
        if step < args.explore_steps:
            adjust_learning_rate(optimizer, step, args.lr, args.lr_decay_rate, args.lr_decay_freq)
        if step>=args.explore_steps:
            if sampling_steps == 0:
                optim_params['lr'] = args.sp_init
                optimizer = utils.get_optim(model.parameters(), args.optim, **optim_params)
            sampling_steps += 1
            adjust_step_size(optimizer, sampling_steps, args.sp_init, args.sp_decay_exp)

        x, y = next(train_sampler)
        if not args.cpu: x, y = x.cuda(), y.cuda()

        start_time = datetime.now()

        model.train()
        ''' variational neural network '''
        _y = model(x)
        lo = - model.log_prior() + F.cross_entropy(_y,y) * model.n
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

            fo_train_loss, fo_train_acc = evaluate(model, forgetted_train_loader, args.cpu)
            logger.info('forgetted train loss {:.2e} \t'
                        'train acc {:.2%}'
                        .format(fo_train_loss, fo_train_acc) )
            utils.add_log(log, 'forgetted_train_loss', fo_train_loss)
            utils.add_log(log, 'forgetted_train_acc', fo_train_acc)

            logger.info('')

    save_checkpoint(args.save_dir, '{}-fin'.format(args.save_name), log, model, optimizer)

    return


if __name__ == '__main__':
    args = get_args()
    logger = utils.generic_init(args)

    try:
        main(args, logger)
    except Exception as e:
        logger.exception('Unexpected exception! %s', e)
