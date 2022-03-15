cd $1

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
