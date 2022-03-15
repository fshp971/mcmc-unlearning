cd $1

root='./exp_data/fmt-lenet'

for i in {1..5}
do
python forget.py \
    --arch lenet \
    --prior-sig 0.1 \
    --dataset fashion-mnist \
    --optim sgld \
    --batch-size 128 \
    --ifs-scaling 0.05 \
    --ifs-iter-T 64 \
    --ifs-samp-T 1 \
    --ifs-rm-bs 64 \
    --rm-idx-path ./data/rm-idx/fmt/6k/$i/rm-idx.pkl \
    --resume-path $root/train/sgld/0k/$i/train-fin-model.pkl \
    --save-dir    $root/remove/sgld/unlearn/6k/$i \
    --save-name unlearn
done
