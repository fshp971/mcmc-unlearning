import pickle
import sys
import logging
import argparse
import numpy as np
import torch
import torch.nn.functional as F

import utils
import models

import kl_est.knn_divergence as knn_div


def get_args():
    parser = argparse.ArgumentParser()
    utils.add_shared_args(parser)

    parser.add_argument('--resume-path1', type=str, default=None)
    parser.add_argument('--resume-path2', type=str, default=None)
    parser.add_argument('--rm-idx-path1', type=str, default=None)
    parser.add_argument('--rm-idx-path2', type=str, default=None)
    parser.add_argument('--sample-num', type=int, default=1)
    parser.add_argument('--sample-dim', type=int, default=None)

    return parser.parse_args()


def get_sample(resume_path, rm_idx_path, args):
    trainset, testset = utils.get_dataset(args.dataset)

    with open(rm_idx_path, 'rb') as f:
        forgetted_idx = pickle.load(f)
    sampler = utils.DataSampler(trainset, args.batch_size)
    sampler.remove( forgetted_idx )

    model = utils.get_mcmc_bnn_arch(args.arch, args.dataset, args.prior_sig)
    model.n = len(sampler)

    if not args.cpu:
        model.cuda()

    optimizer = utils.get_optim(model.parameters(), args.optim,
        lr=args.lr/len(sampler), momentum=args.momentum, weight_decay=args.weight_decay, sghmc_alpha=args.sghmc_alpha)

    state_dict = torch.load(resume_path)
    model.load_state_dict(state_dict['model_state_dict'])
    optimizer.load_state_dict(state_dict['optimizer_state_dict'])

    ''' for backward compatibility '''
    for group in optimizer.param_groups:
        if 'lr_decay' in group:
            group['lr'] *= group['lr_decay']
            group.pop('lr_decay')

    ret = []

    for step in range(args.sample_num):
        x, y = next(sampler)
        if not args.cpu: x, y = x.cuda(), y.cuda()

        model.train()
        lo = -model.log_prior() + F.cross_entropy(model(x), y) * model.n
        optimizer.zero_grad()
        lo.backward()
        optimizer.step()
        with torch.no_grad():
            tmp = []
            for pp in model.parameters():
                if not pp.requires_grad: continue
                tmp.append( pp.data.view(-1) )
            ret.append( torch.cat(tmp).cpu().numpy() )

    ret = np.array(ret)
    return ret


def main(args, logger):
    logger.info('sampling 1')
    sample1 = get_sample(args.resume_path1, args.rm_idx_path1, args)
    print(sample1.shape)

    logger.info('sampling 2')
    sample2 = get_sample(args.resume_path2, args.rm_idx_path2, args)
    print(sample2.shape)

    logger.info('begin')
    kl = knn_div.skl_estimator_efficient(sample1, sample2)
    logger.info('end')

    logger.info('est_kl = {:.3f}'.format(kl) )

if __name__  == '__main__':
    args = get_args()

    fmt = '%(asctime)s %(name)s:%(levelname)s:  %(message)s'
    formatter = logging.Formatter(
        fmt, datefmt='%Y-%m-%d %H:%M:%S')

    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=fmt, datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger()

    # logger.info('Arguments')
    # for arg in vars(args):
    #     logger.info('    {:<22}        {}'.format(arg+':', getattr(args,arg)) )
    # logger.info('')

    try:
        main(args, logger)
    except Exception as e:
        logger.exception('Unexpected exception! %s', e)
