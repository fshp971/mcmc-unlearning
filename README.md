# Knowledge Removal in Sampling-based Bayesian Inference

This is the official repository for ICLR 2022 paper ["Knowledge Removal in Sampling-based Bayesian Inference"](https://openreview.net/forum?id=dTqOcTUOQO) by Shaopeng Fu, Fengxiang He and Dacheng Tao.

## Requirements

- Python 3.8
- PyTorch 1.8.1
- Torchvision 0.9.1

#### Install dependencies using pip

```shell
pip install -r requirements.txt
```

#### Install dependencies using Anaconda

It is recommended to create your experiment environment with [Anaconda3](https://www.anaconda.com/).

```shell
conda install pytorch=1.8.1 torchvision=0.9.1 cudatoolkit=10.2 -c pytorch
```

## Quick Start

To perform MCMC unlearning, you have to first build an `unlearner`. Then, you can remove a batch of datums each time. Here we take MCMC unlearning for Bayesian neural networks (BNNs) as an example.

### Build MCMC unlearning module for Bayesian neural networks

You need to implement the following three class methods:

- `_apply_sample`, which is used to perform MCMC sampling;

- `_fun(self,z)`, which is used to calculate $F(\delta,S)$;
- `_z_fun(self,z)`, which is used to calculate $\sum_{z_j \in S^\prime} h(\delta,z_j)$.

The demo code is as follows:

```python
from mcmc_unlearner import sgmcmcUnlearner

class myUnlearner(sgmcmcUnlearner):
    def _apply_sample(self, z):
        x, y = z
        if not self.cpu: x, y = x.cuda(), y.cuda()
        self.model.train()
        lo = self.model.F(z)
        self.optimizer.zero_grad()
        lo.backward()
        self.optimizer.step()

    def _fun(self, z):
        x, y = z
        if not self.cpu: x, y = x.cuda(), y.cuda()
        self.model.train()
        return self.model.F(z)

    def _z_fun(self, z):
        x, y = z
        if not self.cpu: x, y = x.cuda(), y.cuda()
        self.model.train()
        return self.model.h(z)

unlearner = myUnlearner(model=model, optimizer=optimizer, params=model.parameters(), cpu=False, iter_T=64, scaling=0.1, samp_T=5)
```

where `model` is the Bayesian neural network, `optimizer` is the stochastic gradient MCMC (SGMCMC) sampler, `iter_T` is the number of recursion of calculating the inverse Hessian matrix, and `samp_T` is the number of Monte Carlo sampling times for estimating the expectations in the MCMC influence function.

### Perform MCMC unlearning for Bayesian neural networks

You can remove a batch of datums `[xx,yy]` from your Bayesian neural network as follows:

```python
unlearner.param_dict['scaling'] = init_scaling / remaining_n
unlearner.remove([xx,yy], remaining_sampler)
```

where `remaining_sampler` is a sampler that can repeatedly draw a batch of datums from the current remaining set.

It is recommended to set the scaling factor `scaling` as `init_scaling / remaining_n`, where `init_scaling` is the initial scaling factor, `remaining_n` is the number of the currently remaining datums. Also, you need to adjust `init_scaling` to let the recursive calculation of the inverse Hessian matrix converge.

## Instruction for reproducing results

- For the experiments of Gaussian mixture models (GMMs), please see [./GMM/README.md](./GMM/README.md).
- For the experiments of Bayesian neural networks (BNNs), please see [./BNN/README.md](./BNN/README.md).

## Citation

```
@inproceedings{fu2022knowledge,
  title={Knowledge Removal in Sampling-based Bayesian Inference},
  author={Shaopeng Fu and Fengxiang He and Dacheng Tao},
  booktitle={International Conference on Learning Representations},
  year={2022}
}
```

## Acknowledgment

Part of the code is based on the following repository:

- KL divergence estimators: [https://github.com/nhartland/KL-divergence-estimators](https://github.com/nhartland/KL-divergence-estimators)