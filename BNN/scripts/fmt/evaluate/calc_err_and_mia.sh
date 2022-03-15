cd $1

root='./exp_data/fmt-lenet'
srcidx='0k'
taridx='6k'
method='sgld'

for i in {1..5}
do
python calc_err_and_mia.py \
    --arch lenet \
    --prior-sig 0.1 \
    --optim ${method} \
    --dataset fashion-mnist \
    --batch-size 128 \
    --rm-idx-path ./data/rm-idx/fmt/${taridx}/$i/rm-idx.pkl \
    --resume-path $root/remove/$method/unlearn/${taridx}/$i/unlearn-model.pkl

    # retrain
    # --resume-path $root/train/${method}/${taridx}/$i/train-fin-model.pkl

    # origin
    # --resume-path $root/train/${method}/${srcidx}/$i/train-fin-model.pkl

    # finetune (Importance Sampling)
    # --resume-path $root/remove/$method/finetune/${taridx}/$i/finetune-model.pkl

    # unlearn (MCMC Unlearning)
    # --resume-path $root/remove/$method/unlearn/${taridx}/$i/unlearn-model.pkl
done
