import sys
import logging
import numpy as np
import pickle
import argparse
import torch
import torch.nn.functional as F

import utils


def get_args():
    parser = argparse.ArgumentParser()
    utils.add_shared_args(parser)

    parser.add_argument('--rm-idx-path', type=str, default=None)
    parser.add_argument('--samp-T', type=int, default=1)

    return parser.parse_args()


def get_conf_and_acc(model, loader, cpu):
    conf_vec = []
    acc = utils.AverageMeter()

    model.eval()
    with torch.no_grad():
        for x, y in loader:
            if not cpu: x, y = x.cuda(), y.cuda()
            _y = model(x).softmax(dim=1)
            conf_vec.append( _y.gather(1, y.view(len(y),1)) )
            ac = (_y.argmax(dim=1) == y).sum().item() / len(y)
            acc.update(ac, len(y))

        conf_vec = torch.cat(conf_vec)

    return conf_vec.cpu().numpy(), acc.average()


def main(args, logger):
    trainset, testset = utils.get_dataset(args.dataset)

    with open(args.rm_idx_path, 'rb') as f:
        forgetted_idx = pickle.load(f)
    sampler = utils.DataSampler(trainset, args.batch_size)
    sampler.remove( forgetted_idx )

    train_loader = utils.DataLoader(trainset, args.batch_size)
    train_loader.remove(forgetted_idx)

    forget_loader = utils.DataLoader(trainset, args.batch_size)
    forget_loader.set_sampler_indices(forgetted_idx)

    test_loader = utils.DataLoader(testset, args.batch_size)

    model = utils.get_mcmc_bnn_arch(args.arch, args.dataset, args.prior_sig)
    model.n = len(sampler)

    if not args.cpu:
        model.cuda()

    optimizer = utils.get_optim(model.parameters(), args.optim,
        lr=args.lr/len(sampler), momentum=args.momentum, weight_decay=args.weight_decay, sghmc_alpha=args.sghmc_alpha)

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

    train_acc, forget_acc, test_acc = 0, 0, 0

    ''' mcmc sampling begin '''
    x, y = next(sampler)
    if not args.cpu: x, y = x.cuda(), y.cuda()

    model.eval()
    lo = -model.log_prior() + F.cross_entropy(model(x), y) * model.n
    optimizer.zero_grad()
    lo.backward()
    optimizer.step()
    ''' mcmc sampling end '''

    train_conf,  train_acc  = get_conf_and_acc(model, train_loader,  args.cpu)
    forget_conf, forget_acc = get_conf_and_acc(model, forget_loader, args.cpu)
    test_conf,   test_acc   = get_conf_and_acc(model, test_loader,   args.cpu)

    threshold, std_atk_acc = utils.mia_get_threshold(train_conf, test_conf)
    forget_atk_acc = (forget_conf < threshold).sum() / len(forget_conf)

    logger.info('mia_atk_acc = {:.3%}'.format(forget_atk_acc))
    logger.info('train_err = {:.3%}'.format(1-train_acc)  )
    logger.info('forget_err = {:.3%}'.format(1-forget_acc) )
    logger.info('test_err   = {:.3%}'.format(1-test_acc)   )


if __name__ == '__main__':
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
