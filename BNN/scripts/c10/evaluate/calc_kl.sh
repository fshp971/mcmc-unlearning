cd $1

root='./exp_data/fin/c10-smcnn'
srcidx='0k'
taridx='5k'
method='sgld'


for i in {1..5}
do
python calc_kl.py \
    --arch small-cnn \
    --prior-sig 0.1 \
    --optim ${method} \
    --dataset cifar10 \
    --batch-size 128 \
    --rm-idx-path1 ./data/rm-idx/c10/${taridx}/$i/rm-idx.pkl \
    --resume-path1  $root/remove/${method}/finetune/${taridx}/$i/finetune-model.pkl \
    --rm-idx-path2 ./data/rm-idx/c10/${taridx}/$i/rm-idx.pkl \
    --resume-path2  $root/train/${method}/${taridx}/$i/train-fin-model.pkl \
    --sample-num 100

    # origin
    # --rm-idx-path1 ./data/rm-idx/c10/${srcidx}/$i/rm-idx.pkl \
    # --resume-path1  $root/train/${method}/${srcidx}/$i/train-fin-model.pkl \

    # finetune (Importance Sampling)
    # --rm-idx-path1 ./data/rm-idx/c10/${taridx}/$i/rm-idx.pkl \
    # --resume-path1  $root/remove/${method}/finetune/${taridx}/$i/finetune-model.pkl \

    # unlearn (MCMC Unlearning)
    # --rm-idx-path1 ./data/rm-idx/c10/${taridx}/$i/rm-idx.pkl \
    # --resume-path1  $root/remove/${method}/unlearn/${taridx}/$i/unlearn-model.pkl \
done
