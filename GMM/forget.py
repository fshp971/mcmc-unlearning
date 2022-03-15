from datetime import datetime
import os
import pickle
import argparse
import numpy as np
import torch
import torch.nn.functional as F

# import bayes_forgetters
from mcmc_unlearner import sgmcmcUnlearner
import utils
import models


class myUnlearner(sgmcmcUnlearner):
    def _apply_sample(self, z):
        x = z[0]
        if not self.cpu: x = x.cuda()
        lo = self.model.loss(x, self.model(x))
        self.optimizer.zero_grad()
        lo.backward()
        self.optimizer.step()

    def _fun(self, z):
        x = z[0]
        if not self.cpu: x = x.cuda()
        self.model.train()
        return self.model.loss(x, self.model(x))

    def _z_fun(self, z):
        x = z[0]
        if not self.cpu: x = x.cuda()
        self.model.train()
        return self.model.x_loss(x, self.model(x))


def get_args():
    parser = argparse.ArgumentParser()
    utils.add_shared_gmm_args(parser)

    return parser.parse_args()


def forget_eval_one_time(model, train_loader, forgetted_train_loader, log, cpu):
    remain_train_loss, remain_train_log_pp = utils.evaluate(model, train_loader, cpu)
    forgetted_train_loss, forgetted_train_log_pp = utils.evaluate(model, forgetted_train_loader, cpu)
    utils.add_log(log, 'remain_train_loss', remain_train_loss)
    utils.add_log(log, 'remain_train_log_pp', remain_train_log_pp)
    utils.add_log(log, 'forgetted_train_loss', forgetted_train_loss)
    utils.add_log(log, 'forgetted_train_log_pp', forgetted_train_log_pp)
    logger.info('remaining train loss {:.2e} \t train log pp {:.2e}'
                .format(remain_train_loss, remain_train_log_pp))
    logger.info('forgetted train loss {:.2e} \t train log pp {:.2e}'
                .format(forgetted_train_loss, forgetted_train_log_pp))


def save_checkpoint(save_dir, save_name, log, model, optimizer):
    with open('{}/{}-log.pkl'.format(save_dir, save_name), 'wb') as f:
        pickle.dump(log, f)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        }, '{}/{}-model.pkl'.format(save_dir, save_name))


def main(args, logger):
    trainset = utils.get_GMM2d()
    forgetted_idx = utils.get_forget_idx(trainset, args.ifs_kill_num)

    forgetted_idx_loader = utils.IndexBatchSampler(
        batch_size=args.ifs_rm_bs, indices=forgetted_idx)

    train_sampler = utils.DataSampler(trainset, args.batch_size)

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

    unlearner = myUnlearner(
        model     = model,
        optimizer = optimizer,
        params    = model.parameters(),
        cpu       = args.cpu,
        iter_T    = args.ifs_iter_T,
        scaling   = args.ifs_scaling,
        samp_T    = args.ifs_samp_T,)

    log = dict()
    log['user_time'] = 0
    utils.add_log(log, 'forgetted_idx', forgetted_idx)

    forget_eval_one_time(model, train_loader, forgetted_train_loader, log, args.cpu)
    logger.info('')

    for ii in forgetted_idx_loader:
        xx, yy = trainset[ii]
        xx, yy = torch.Tensor(xx), torch.Tensor(yy)

        scaling = args.ifs_scaling / len(train_sampler)
        unlearner.param_dict['scaling'] = scaling

        ''' start calculation of time '''
        start_time = datetime.now()

        unlearner.remove([xx,yy], train_sampler)

        if not args.cpu:
            torch.cuda.synchronize()

        end_time = datetime.now()
        user_time = (end_time - start_time).total_seconds()
        ''' end calculation of time '''

        log['user_time'] += user_time

        train_sampler.remove(ii)

        ''' after removal, update the number of remaining datums '''
        unlearner.model.n = len(train_sampler)

        logger.info('remaining trainset size {}'.format(len(train_sampler)))
        logger.info('user time {:.3f} sec \t'
                    'cumulated user time {:.3f} mins'
                    .format(user_time, log['user_time']/60) )

        forget_eval_one_time(model, train_loader, forgetted_train_loader, log, args.cpu)
        logger.info('')

    utils.save_checkpoint(args.save_dir, args.save_name, log, model, optimizer)

    return


if __name__ == '__main__':
    args = get_args()
    logger = utils.generic_init(args)

    try:
        main(args, logger)
    except Exception as e:
        logger.exception('Unexpected exception! %s', e)
