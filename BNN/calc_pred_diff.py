import sys
import logging
import numpy as np
import pickle
import argparse
import torch
import torch.nn.functional as F

import utils

import os


def get_args():
    parser = argparse.ArgumentParser()
    utils.add_shared_args(parser)

    parser.add_argument('--resume-dir1',  type=str, default=None)
    parser.add_argument('--resume-name1', type=str, default=None)

    parser.add_argument('--resume-dir2',  type=str, default=None)
    parser.add_argument('--resume-name2', type=str, default=None)

    parser.add_argument('--rm-idx-dir1', type=str, default=None)
    parser.add_argument('--rm-idx-dir2', type=str, default=None)

    parser.add_argument('--samp-T', type=int, default=5)

    return parser.parse_args()


def get_conf_and_acc(model, loader, cpu):
    conf_vec = []
    acc = utils.AverageMeter()

    model.eval()
    with torch.no_grad():
        for x, y in loader:
            if not cpu: x, y = x.cuda(), y.cuda()
            _y = model(x).softmax(dim=1)
            # conf_vec.append( _y.gather(1, y.view(len(y),1)) )
            conf_vec.append(_y)
            ac = (_y.argmax(dim=1) == y).sum().item() / len(y)
            acc.update(ac, len(y))

        conf_vec = torch.cat(conf_vec)

    return conf_vec.cpu().numpy(), acc.average()


def get_confs(model, loaders, args):
    confs = {}

    for name in ['train', 'forget', 'test']:
        conf, _ = get_conf_and_acc(model, loaders[name], args.cpu)
        confs[name] = conf

    return confs


def get_loaders(trainset, testset, batch_size=128, samp_rm_idx_path=None, tar_rm_idx_path=None):
    loaders = {}

    with open(samp_rm_idx_path, 'rb') as f:
        samp_forgetted_idx = pickle.load(f)

    loaders['sample'] = utils.DataSampler(trainset, batch_size)
    loaders['sample'].remove( samp_forgetted_idx )

    with open(tar_rm_idx_path, 'rb') as f:
        tar_forgetted_idx = pickle.load(f)

    loaders['train'] = utils.DataLoader(trainset, batch_size)
    loaders['train'].remove(tar_forgetted_idx)

    loaders['forget'] = utils.DataLoader(trainset, batch_size)
    loaders['forget'].set_sampler_indices(tar_forgetted_idx)

    loaders['test'] = utils.DataLoader(testset, batch_size)

    return loaders


def get_results(trainset, testset, samp_rm_idx_dir, tar_rm_idx_dir, resume_dir, resume_name, args):
    model = utils.get_mcmc_bnn_arch(args.arch, args.dataset, args.prior_sig)

    if not args.cpu:
        model.cuda()

    confs = {'train':[], 'forget':[], 'test':[]}
    for i in range(1, args.samp_T+1):
        ''' restore data loaders '''
        samp_rm_idx_path = os.path.join(samp_rm_idx_dir, '{}'.format(i), 'rm-idx.pkl')
        tar_rm_idx_path = os.path.join(tar_rm_idx_dir, '{}'.format(i), 'rm-idx.pkl')
        loaders = get_loaders(trainset, testset, args.batch_size, samp_rm_idx_path, tar_rm_idx_path)

        ''' restore model / sampler '''
        resume_path = os.path.join(resume_dir, '{}'.format(i), resume_name)
        state_dict = torch.load(resume_path)
        model.load_state_dict(state_dict['model_state_dict'])
        model.n = len(loaders['sample'])

        temp_confs = get_confs(model, loaders, args)
        for key in confs.keys():
            confs[key].append(temp_confs[key])

    return confs


def main(args, logger):
    trainset, testset = utils.get_dataset(args.dataset)

    confs_1 = get_results(trainset, testset, args.rm_idx_dir1, args.rm_idx_dir2, args.resume_dir1, args.resume_name1, args)
    confs_2 = get_results(trainset, testset, args.rm_idx_dir2, args.rm_idx_dir2, args.resume_dir2, args.resume_name2, args)

    for name in ['train', 'forget', 'test']:
        res = []

        for i in range(args.samp_T):
            res.append( np.abs(confs_1[name][i] - confs_2[name][i]).sum(axis=1).mean().item() )

        res = np.array(res)

        print('pred_diff: {}: mean={:.3f}, std={:.3f}'.format(name, res.mean(), res.std()) )


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
