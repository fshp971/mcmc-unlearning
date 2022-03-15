cd $1

root='./exp_data/fmt-lenet'
srcidx='0k'
taridx='6k'
method='sgld'


python calc_pred_diff.py \
    --arch lenet \
    --prior-sig 0.1 \
    --dataset fashion-mnist \
    --batch-size 128 \
    --rm-idx-dir1   ./data/rm-idx/fmt/${taridx} \
    --resume-dir1   $root/remove/${method}/unlearn/${taridx} \
    --resume-name1  unlearn-model.pkl \
    --rm-idx-dir2   ./data/rm-idx/fmt/${taridx} \
    --resume-dir2   $root/train/${method}/${taridx} \
    --resume-name2  train-fin-model.pkl \
    --samp-T 5

    # origin
    # --rm-idx-dir1   ./data/rm-idx/fmt/${srcidx} \
    # --resume-dir1   $root/train/${method}/${srcidx} \
    # --resume-name1  train-fin-model.pkl \

    # finetune
    # --rm-idx-dir1   ./data/rm-idx/fmt/${taridx} \
    # --resume-dir1   $root/remove/${method}/finetune/${taridx} \
    # --resume-name1  finetune-model.pkl \

    # unlearn
    # --rm-idx-dir1   ./data/rm-idx/fmt/${taridx} \
    # --resume-dir1   $root/remove/${method}/unlearn/${taridx} \
    # --resume-name1  unlearn-model.pkl \
