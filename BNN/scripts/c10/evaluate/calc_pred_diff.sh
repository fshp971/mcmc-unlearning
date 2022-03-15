cd $1

root='./exp_data/c10-smcnn'
srcidx='0k'
taridx='5k'
method='sgld'


python calc_pred_diff.py \
    --arch small-cnn \
    --prior-sig 0.1 \
    --dataset cifar10 \
    --batch-size 128 \
    --rm-idx-dir1   ./data/rm-idx/c10/${taridx} \
    --resume-dir1   $root/remove/${method}/unlearn/${taridx} \
    --resume-name1  unlearn-model.pkl \
    --rm-idx-dir2   ./data/rm-idx/c10/${taridx} \
    --resume-dir2   $root/train/${method}/${taridx} \
    --resume-name2  train-fin-model.pkl \
    --samp-T 5

    # origin
    # --rm-idx-dir1   ./data/rm-idx/c10/${srcidx} \
    # --resume-dir1   $root/train/${method}/${srcidx} \
    # --resume-name1  train-fin-model.pkl \

    # finetune
    # --rm-idx-dir1   ./data/rm-idx/c10/${taridx} \
    # --resume-dir1   $root/remove/${method}/finetune/${taridx} \
    # --resume-name1  finetune-model.pkl \

    # unlearn
    # --rm-idx-dir1   ./data/rm-idx/c10/${taridx} \
    # --resume-dir1   $root/remove/${method}/unlearn/${taridx} \
    # --resume-name1  unlearn-model.pkl \
