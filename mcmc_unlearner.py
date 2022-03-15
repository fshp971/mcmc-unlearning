import torch


class sgmcmcUnlearner():
    ''' Implementation of SGMCMC unlearner.

    It will first calculate the following function I(target):

        I(target) = -H_theta^(-1) @ G_theta

    and then remove the influence from model parameter as follow:

        theta_new = theta + (-1) * I(target)

    where theta is the model parameter, H_theta is the hessian matrix
    of F(theta, S) (see the mcmc influence function), and G_theta is the
    gradient of h(theta, z) (see the mcmc influence function).

    The calculation of I(target) relies on Taylor series (for calculating
    hessian-inverse) and efficient hessian-vector product, just as the
    following references suggest.

    References:
        [1] https://arxiv.org/pdf/1602.03943.pdf
        [2] https://arxiv.org/pdf/1703.04730.pdf

    Args:
        model (torch.nn.Module): the Bayesian model.

        optimizer (torch.optim): sgmcmc optimizer that used to
            sample (update) model parameter (i.e., params) for
            estimating the expectation terms in I(target).

        params (iterable): iterable of parameters to optimize.

        cpu (bool): True=use cpu, False=use cuda.

        iter_T (int): the number of iterations that computing
            a single hessian-inverse

        scaling (float): The algorithm for calculating hessian-inverse
            converges iff |det(H)| <= 1, and `scaling` is provided to
            let hessian matrix H meet that condition. Specifically, in
            each iteration, the following inverse matrix will be calculated,
                (scaling * H)^(-1) = (1/scaling) * H^(-1),
            and once the whole iterations finished, the obtained result
            will be rescaled to its true value.

        samp_T (int): the number of times of SGMCMC sampling for
            estimating the expectation terms in I(target).
    '''
    def __init__(self, model, optimizer, params, cpu, iter_T, scaling, samp_T):
        self.model = model
        self.optimizer = optimizer
        self.params = []
        for pp in params:
            if pp.requires_grad: self.params.append(pp)
        # self.params = [pp for pp in params]
        self.cpu = cpu
        self.param_dict = dict(iter_T=iter_T, scaling=scaling, samp_T=samp_T)

    def _fun(self, z):
        ''' Calculates F(theta, S).

        Args:
            z (torch.Tensor): a batch of datums sampled from the
                current remaining set.
        '''
        raise NotImplementedError('you should implement `bifForgetter._fun` yourself before using it')

    def _z_fun(self, z):
        ''' Calculates h(theta, z).

        Args:
            z (torch.Tensor): datums that going to be removed.
        '''
        raise NotImplementedError('you should manually implement `bifForgetter._z_fun` yourself before using it')

    def _z_fun_grad(self, z):
        ''' Calculates the the gradient of h(theta, z).

        Args:
            z (torch.Tensor): datums that going to be removed.
        '''
        z_lo = self._z_fun(z)
        return torch.autograd.grad(z_lo, self.params)

    def _hvp(self, hh, zz):
        ''' Calculates the hessian-vector product.
        Given: `hh`, `zz`; return: (scaling * H(zz)) @ hh.

        Args:
            hh (torch.Tensor): the intermediate result (a vector) when
                iteratively computing inverse-hessian-vector product.

            zz (torch.tensor): a batch of datums sampled from
                the current remaining set.
        '''
        lo = self._fun(zz)
        lo *= self.param_dict['scaling']
        tmp = torch.autograd.grad(lo, self.params, create_graph=True)

        lloo = 0
        for hg, pg in zip(hh, tmp):
            lloo += (hg*pg).sum()
        tmp = torch.autograd.grad(lloo, self.params)

        return tmp

    def _apply_sample(self, z):
        ''' Conducts SGMCMC sampling for one step (for estimating the
        expectation terms in I(target)).

        Args:
            z (torch.Tensor): a batch of datums sampled from the current
                remaining set.
        '''
        raise NotImplementedError('you should implement `sgmcmcForgetter._apply_sample` yourself before using it')

    def remove(self, target, sampler):
        ''' main function to conduct mcmc unlearning

        Args:
            target (torch.Tensor): the to be removed datums.

            sampler (iterable): iterable that can repeatly sample
                a batch of datums from the current remaining set.
        '''

        ''' first, estimates the expectation of gradient G_theta(target)
        via Monte Carlo method
        '''
        v_grad = None
        for i in range( self.param_dict['samp_T'] ):
            self._apply_sample( next(sampler) )
            tmp = self._z_fun_grad(target)
            if v_grad is None: v_grad = tmp
            else:
                for vg, pg in zip(v_grad, tmp): vg += pg
        for vg in v_grad: vg /= self.param_dict['samp_T']

        ''' next, repeatedly compute H_theta(S)^(-1) @ G,
        formula: hh_{t+1} = v_grad + (I - H(zz_t)) @ hh_t
                          = v_grad + hh_t - H(zz_t) @ hh_t
        '''
        hh = [vg.clone() for vg in v_grad]
        for tt in range( self.param_dict['iter_T'] ):
            ''' estimates the expectation of hvp H_theta @ G '''
            tmp = None
            for ii in range( self.param_dict['samp_T'] ):
                ''' estimates the expectation via Monte Carlo method '''
                self._apply_sample( next(sampler) )
                zz = next(sampler)
                ''' hessian-vector product '''
                rep_tmp = self._hvp(hh, zz)
                if tmp is None: tmp = rep_tmp
                else:
                    for tp, rtp in zip(tmp, rep_tmp):
                        tp += rtp
            for tp in tmp: tp /= self.param_dict['samp_T']

            for hg, vg, pg in zip(hh, v_grad, tmp):
                hg += vg - pg

        target_grad = hh

        ''' re-scaling: (scaling * H^(-1))v  =>  H^(-1)v '''
        for tg in target_grad:
            tg *= - self.param_dict['scaling']

        ''' apply the forgetting-gradient '''
        for pp, tg in zip(self.params, target_grad):
            pp.data -= tg
