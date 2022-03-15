import pickle
import argparse
import os
import numpy as np


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--type', type=str, default='special',
    choices=['origin', 'special'])
    parser.add_argument('--gmm-gen-kk', type=int, default=4)
    parser.add_argument('--gmm-gen-std', type=float, default=2)
    parser.add_argument('--gmm-gen-n', type=int, default=2000)

    parser.add_argument('--random-seed', type=int, default=23498353)
    parser.add_argument('--save-dir', type=str, default='./data/GMMs')
    parser.add_argument('--save-name', type=str, default='gmm-2d-syn-set.pkl')

    return parser.parse_args()


def gen_data_special(kk, n, prior_std):
    np.random.seed(23498353)
    # dx1 = np.random.rand() * 3
    # dx2 = np.random.rand() * 3
    dx1 = prior_std[0] ** 2
    dx2 = prior_std[1] ** 2
    coef = np.random.rand() * np.sqrt(dx1) * np.sqrt(dx2)
    cc = np.array([[dx1, coef], [coef, dx2]])
    mu = np.random.multivariate_normal([0,0], cc, size=kk)

    # print(mu.shape)
    # print(mu)
    mu[1] += [-1,2.5]
    mu[3] += [1,-0]

    cov = []
    for k in range(kk):
        if k==0:
            dx1 = 1
            dx2 = 0.05
            cov.append( np.array([[dx1, 0], [0, dx2]]) )
        elif k==2:
            dx1 = 0.05
            dx2 = 1
            cov.append( np.array([[dx1, 0], [0, dx2]]) )
        else:
            dx1 = 1
            dx2 = 1
            cov.append( np.array([[dx1, 0], [0, dx2]]) )

    x, y = [], []
    for i in range(n):
        k = np.random.randint(kk)
        x.append( np.random.multivariate_normal(mu[k], cov[k]) )
        y.append(k)

    x = np.array(x).astype(np.float32)
    y = np.array(y).astype(np.int)

    return x, y


def gen_data_origin(kk, n, prior_std):
    # dx1 = np.random.rand() * 3
    # dx2 = np.random.rand() * 3
    dx1 = prior_std[0] ** 2
    dx2 = prior_std[1] ** 2
    coef = np.random.rand() * np.sqrt(dx1) * np.sqrt(dx2)
    cc = np.array([[dx1, coef], [coef, dx2]])
    mu = np.random.multivariate_normal([0,0], cc, size=kk)

    cov = []
    for k in range(kk):
        # dx1 = np.random.rand() * (prior_std[0]**2) * 0.2
        # dx2 = np.random.rand() * (prior_std[1]**2) * 0.2
        dx1 = 1
        dx2 = 1
        # coef = np.random.rand() * np.sqrt(dx1) * np.sqrt(dx2)
        coef = 0
        cov.append( np.array([[dx1, coef], [coef, dx2]]) )

    x, y = [], []
    for i in range(n):
        k = np.random.randint(kk)
        x.append( np.random.multivariate_normal(mu[k], cov[k]) )
        y.append(k)

    x = np.array(x).astype(np.float32)
    y = np.array(y).astype(np.int)

    return x, y


def main():
    args = get_args()

    ''' fix random seed '''
    # SEED = 19260817
    # # NP_SEED = 10**9 + 9
    # NP_SEED = 31894921
    np.random.seed(args.random_seed)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    gen_data_params = {
        'kk': args.gmm_gen_kk,
        'n': args.gmm_gen_n,
        'prior_std': [args.gmm_gen_std, args.gmm_gen_std],
    }

    if args.type == 'origin':
        x, y = gen_data_origin(**gen_data_params)
    elif args.type == 'special':
        x, y = gen_data_special(**gen_data_params)
    else:
        raise ValueError

    save_path = '{}/{}'.format(args.save_dir, args.save_name)
    with open(save_path, 'wb') as f:
        pickle.dump({'x':x, 'y':y}, f)

    print('the generated dataset has been saved to `{}`'.format(save_path))

    return


if __name__ == '__main__':
    main()
