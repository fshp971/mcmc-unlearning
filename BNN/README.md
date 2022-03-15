# Experiments on Bayesian Neural Networks

This folder contains the code and scripts for the Bayesian neural network (BNN) experiments.

## Quick Start

We give an example of MCMC unlearning with SGLD on CIFAR-10 dataset. Other experiment scripts can be found in [./scripts](./scripts).

#### Step 1: Prepare pretrained model

```shell
python prepare_model.py --arch small-cnn --pretrain-dataset cifar100 --finetune-dataset cifar10 --optim sgd --momentum 0.9 --batch-size 128 --lr 0.1 --burn-in-steps 10000 --save-freq 20000 --eval-freq 2000 --save-dir ./exp_data/c10-smcnn/premodel/sgld/1 --save-name premodel
```

#### Step 2: Train models

- Train the origin model on the full training set:

```shell
python train.py --arch small-cnn --prior-sig 0.1 --dataset cifar10 --burn-in-steps 20000 --optim sgld --batch-size 128 --sp-init 0.01 --sp-decay-exp -0.5005 --lr 0.01 --lr-decay-rate 1 --lr-decay-freq 2000 --explore-steps 5000 --eval-freq 2000 --rm-idx-path ./data/rm-idx/c10/0k/1/rm-idx.pkl --init-model-path ./exp_data/c10-smcnn/premodel/sgld/1/premodel-fin-model.pkl --save-dir ./exp_data/c10-smcnn/train/sgld/0k/1 --save-name train
```

- Train the target model on the remaining set:

```shell
python train.py --arch small-cnn --prior-sig 0.1 --dataset cifar10 --burn-in-steps 20000 --optim sgld --batch-size 128 --sp-init 0.01 --sp-decay-exp -0.5005 --lr 0.01 --lr-decay-rate 1 --lr-decay-freq 2000 --explore-steps 5000 --eval-freq 2000 --rm-idx-path ./data/rm-idx/c10/5k/1/rm-idx.pkl --init-model-path ./exp_data/c10-smcnn/premodel/sgld/1/premodel-fin-model.pkl --save-dir ./exp_data/c10-smcnn/train/sgld/5k/1 --save-name train
```

#### Step 3: Perform machine unlearning

- Perform via MCMC unlearning method:

```shell
python forget.py --arch small-cnn --prior-sig 0.1 --dataset cifar10 --optim sgld --batch-size 128 --ifs-scaling 0.08 --ifs-iter-T 64 --ifs-samp-T 1 --ifs-rm-bs 64 --rm-idx-path ./data/rm-idx/c10/5k/1/rm-idx.pkl --resume-path ./exp_data/c10-smcnn/train/sgld/0k/1/train-fin-model.pkl --save-dir ./exp_data/c10-smcnn/remove/sgld/unlearn/5k/1 --save-name unlearn
```

- Perform via importance sampling method:

```shell
python finetune.py --arch small-cnn --prior-sig 0.1 --dataset cifar10 --optim sgld --batch-size 128 --ifs-rm-bs 64 --finetune-steps 1000 --rm-idx-path ./data/rm-idx/c10/5k/1/rm-idx.pkl --resume-path ./exp_data/c10-smcnn/train/sgld/0k/1/train-fin-model.pkl --save-dir ./exp_data/c10-smcnn/rm/sgld/finetune-1k/5k/1 --save-name finetune
```

#### Step 4: Evaluate the performance of unlearning

- Calculate classification errors and the membership inference attack accuracy:

```shell
python calc_err_and_mia.py --arch small-cnn --prior-sig 0.1 --optim sgld --dataset cifar10 --batch-size 128 --rm-idx-path ./data/rm-idx/c10/5k/1/rm-idx.pkl --resume-path ./exp_data/c10-smcnn/remove/sgld/unlearn/5k/1/unlearn-model.pkl
```

- Calculate the knowledge removal estimator:

```shell
python calc_kl.py --arch small-cnn --prior-sig 0.1 --optim sgld --dataset cifar10 --batch-size 128 --rm-idx-path1 ./data/rm-idx/c10/5k/1/rm-idx.pkl --resume-path1 ./exp_data/fin/c10-smcnn/remove/sgld/finetune/5k/1/finetune-model.pkl --rm-idx-path2 ./data/rm-idx/c10/5k/1/rm-idx.pkl --resume-path2 ./exp_data/fin/c10-smcnn/train/sgld/5k/1/train-fin-model.pkl --sample-num 100
```

- Calculate the prediction difference:

```shell
python calc_pred_diff.py --arch small-cnn --prior-sig 0.1 --dataset cifar10 --batch-size 128 --rm-idx-dir1 ./data/rm-idx/c10/5k --resume-dir1 ./exp_data/c10-smcnn/remove/sgld/unlearn/5k --resume-name1 unlearn-model.pkl --rm-idx-dir2 ./data/rm-idx/c10/5k --resume-dir2 ./exp_data/c10-smcnn/train/sgld/5k --resume-name2 train-fin-model.pkl --samp-T 5
```