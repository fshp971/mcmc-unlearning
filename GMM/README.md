# Experiments on Gaussian Mixture Models

This folder contains the code and scripts for the Gaussian mixture model (GMM) experiments.

## Quick Start

We give an example of MCMC unlearning with SGLD. Other experiment scripts can be found in [./scripts](./scripts).

#### Step 1: Train the GMM on the full dataset

```shell
python train.py \
    --cpu \
    --gmm-kk 4 \
    --gmm-std 1 \
    --optim sgld \
    --batch-size 64 \
    --burn-in-steps 4000 \
    --eval-freq 100 \
    --lr 4 \
    --lr-decay-exp -0.5005 \
    --ifs-scaling 1 \
    --ifs-iter-T 32 \
    --ifs-samp-T 5 \
    --ifs-iter-bs 64 \
    --ifs-rm-bs 4 \
    --ifs-kill-num 0 \
    --samp-num 4000 \
    --save-dir ./exp_data/gmm/sgld \
    --save-name full
```

#### Step 2: Remove datums from the trained GMM

```shell
python forget.py \
    --cpu \
    --gmm-kk 4 \
    --gmm-std 1 \
    --optim sgld \
    --batch-size 64 \
    --burn-in-steps 2000 \
    --eval-freq 100 \
    --lr 4 \
    --lr-decay-exp -0.5005 \
    --ifs-scaling 1 \
    --ifs-iter-T 32 \
    --ifs-samp-T 5 \
    --ifs-iter-bs 64 \
    --ifs-rm-bs 4 \
    --ifs-kill-num 800 \
    --resume-path ./exp_data/gmm/sgld/full-model.pkl \
    --save-dir ./exp_data/gmm/sgld \
    --save-name forget
```

