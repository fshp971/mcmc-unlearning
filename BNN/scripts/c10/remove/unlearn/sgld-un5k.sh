cd $1

root='./exp_data/c10-smcnn'

for i in {1..5}
do
python forget.py \
    --arch small-cnn \
    --prior-sig 0.1 \
    --dataset cifar10 \
    --optim sgld \
    --batch-size 128 \
    --ifs-scaling 0.08 \
    --ifs-iter-T 64 \
    --ifs-samp-T 1 \
    --ifs-rm-bs 64 \
    --rm-idx-path ./data/rm-idx/c10/5k/$i/rm-idx.pkl \
    --resume-path $root/train/sgld/0k/$i/train-fin-model.pkl \
    --save-dir    $root/remove/sgld/unlearn/5k/$i \
    --save-name unlearn
done
