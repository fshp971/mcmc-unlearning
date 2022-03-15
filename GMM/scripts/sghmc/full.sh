cd $1

python train.py \
    --cpu \
    --gmm-kk 4 \
    --gmm-std 1 \
    --optim sghmc \
    --sghmc-alpha 0.9 \
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
    --save-dir ./exp_data/gmm/sghmc \
    --save-name full
