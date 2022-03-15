import argparse


def add_shared_gmm_args(parser):
    assert isinstance(parser, argparse.ArgumentParser)

    ''' parameters for GMM '''
    parser.add_argument('--gmm-kk', type=int, default=5,
                        help='set the number of the cluster centers of GMM')
    parser.add_argument('--gmm-std', type=float, default=5,
                        help='set the prior deviation of GMM')

    parser.add_argument('--ifs-scaling', type=float, default=1e-5,
                        help='set the initial scaling factor of calculating the inverse Hessian matrix')
    parser.add_argument('--ifs-iter-T', type=int, default=100,
                        help='set the number of recursion of calculating the inverse Hessian matrix')
    parser.add_argument('--ifs-samp-T', type=int, default=1,
                        help='set the number of sampling times for estimating the expectations in the MCMC influence function')
    parser.add_argument('--ifs-iter-bs', type=int, default=128,
                        help='set the batch size of drawing training datums during forgetting, usually as same as the batch size during training')
    parser.add_argument('--ifs-rm-bs', type=int, default=1,
                        help='set the removing batch size during forgetting')
    parser.add_argument('--ifs-kill-num', type=int, default=1000,
                        help='set the number of datums to be removed')

    ''' parameters for optimizers / samplers '''
    parser.add_argument('--optim', type=str, default='sgd',
                        choices=['sgd', 'adam', 'sgld', 'sghmc'],
                        help='select which optimizer to use')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='set the batch size during training, evaluating, and sampling (sampling is only in exp with SGMCMC)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='set the initial learning rate')
    parser.add_argument('--lr-decay-exp', type=float, default=None,
                        help='if not None, set the learning rate schedule as e^({lr-decay-exp}*t), where t is the training iteration step')
    parser.add_argument('--lr-decay-rate', type=float, default=None,
                        help='if not None, set the learning rate decay rate')
    parser.add_argument('--lr-decay-freq', type=int, default=None,
                        help='if not None, set the decay frequency of learning rate')
    parser.add_argument('--weight-decay', type=float, default=0,
                        help='set the weight decay, usually 0')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='set the momentum of SGD, usually 0')
    parser.add_argument('--sghmc-alpha', type=float, default=1.0,
                        help='set the alpha factor of SGHMC')

    parser.add_argument('--burn-in-steps', type=int, default=10000,
                        help='set the number of training iterations')
    parser.add_argument('--eval-freq', type=int, default=100,
                        help='frequency of evaluating during training')
    parser.add_argument('--cpu', action='store_true',
                        help='select to use cpu, otherwise use gpu')

    parser.add_argument('--save-dir', type=str, default='./save_temp',
                        help='set which dictionary to save the experiment result')
    parser.add_argument('--save-name', type=str, default=None,
                        help='set the save name of the experiment result')

    parser.add_argument('--resume-path', type=str, default=None,
                        help='set which path to restore the model and optimizer')
