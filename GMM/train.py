from datetime import datetime
import pickle
import argparse
import logging
import numpy as np
import torch

import models
import utils


def get_args():
    parser = argparse.ArgumentParser()
    utils.add_shared_gmm_args(parser)

    parser.add_argument('--samp-num', type=int, default=0)

    return parser.parse_args()


def mcmc_sampling_points(model, optimizer, samp_num, sampler, cpu):
    points = []

    model.train()
    for i in range(samp_num):
        x, _ = next(sampler)
        if not cpu: x = x.cuda()
        _y = model(x)
        loss = model.loss(x, _y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        points.append( model.dump_numpy_dict()['mk'].copy() )

    points = np.array(points)

    return points


def save_checkpoint(save_dir, save_name, log, model, optimizer, points=None, save_cluster=True):
    with open('{}/{}-log.pkl'.format(save_dir, save_name), 'wb') as f:
        pickle.dump(log, f)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        }, '{}/{}-model.pkl'.format(save_dir, save_name))

    if points is not None:
        with open('{}/{}-samp-pts.pkl'.format(save_dir, save_name),'wb') as f:
            pickle.dump(points, f)

    if save_cluster:
        with open('{}/{}-cluster.pkl'.format(save_dir, save_name), 'wb') as f:
            pickle.dump(model.dump_numpy_dict(), f)


def main(args, logger):
    trainset = utils.get_GMM2d()
    forgetted_idx = utils.get_forget_idx(trainset, args.ifs_kill_num)

    train_sampler = utils.DataSampler(trainset, args.batch_size)
    train_sampler.remove(forgetted_idx)

    train_loader = utils.DataLoader(trainset, args.batch_size)
    train_loader.remove(forgetted_idx)

    forgetted_train_loader = utils.DataLoader(trainset, args.batch_size)
    forgetted_train_loader.set_sampler_indices(forgetted_idx)

    ''' end of retrieving data '''

    model = models.GMM(kk=args.gmm_kk, dim=2, std=[args.gmm_std, args.gmm_std], n=None)

    if not args.cpu:
        model.cuda()

    args.lr /= len(trainset)
    optimizer = utils.get_optim(model.parameters(), args.optim,
        lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, sghmc_alpha=args.sghmc_alpha)

    model.n = len(train_sampler)

    log = dict()
    log['user_time'] = 0

    for step in range(args.burn_in_steps):
        decay = 1.0
        if args.lr_decay_exp is not None:
            decay = (step+1) ** args.lr_decay_exp
        elif args.lr_decay_rate is not None:
            decay = args.lr_decay_rate ** (step // args.lr_decay_freq)

        for group in optimizer.param_groups:
            group['lr'] = args.lr * decay

        z = next(train_sampler)
        if not args.cpu: x, y = x.cuda(), y.cuda()

        start_time = datetime.now()

        model.train()
        x, _ = z

        if not args.cpu: x = x.cuda()

        _y = model(x)
        loss = model.loss(x, _y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        log_pp = _y.max(dim=1)[0].data.log().mean()

        if not args.cpu:
            torch.cuda.synchronize()
        end_time = datetime.now()
        user_time = (end_time - start_time).total_seconds()
        log['user_time'] += user_time

        utils.add_log(log, 'burn_in_loss',   loss.item())
        utils.add_log(log, 'burn_in_log_pp', log_pp.item())

        if (step+1) % args.eval_freq == 0:
            logger.info('burn-in step [{}/{}]:'
                        .format(step+1, args.burn_in_steps))
            logger.info('user time {:.3f} sec \t'
                        'cumulated user time {:.3f} mins'
                        .format(user_time, log['user_time']/60) )

            logger.info('burn-in loss {:.2e} \t'
                        'burn-in log_pp {:.2e}'
                        .format(loss, log_pp) )

            fo_train_loss, fo_train_log_pp = utils.evaluate(model, forgetted_train_loader, args.cpu)
            logger.info('forgetted train loss {:.2e} \t'
                        'train log pp {:.2e}'
                        .format(fo_train_loss, fo_train_log_pp) )
            utils.add_log(log, 'forgetted_train_loss', fo_train_loss)
            utils.add_log(log, 'forgetted_train_log_pp', fo_train_log_pp)

            logger.info('')

    points = mcmc_sampling_points(model, optimizer, args.samp_num, train_sampler, args.cpu)
    save_checkpoint(args.save_dir, args.save_name, log, model, optimizer, points)
    

if __name__ == '__main__':
    args = get_args()
    logger = utils.generic_init(args)

    try:
        main(args, logger)
    except Exception as e:
        logger.exception('Unexpected exception! %s', e)
