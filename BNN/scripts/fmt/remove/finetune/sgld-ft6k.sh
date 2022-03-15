cd $1

root='./exp_data/fmt-lenet'

for i in {1..5}
do
python finetune.py \
    --arch lenet \
    --prior-sig 0.1 \
    --dataset fashion-mnist \
    --optim sgld \
    --batch-size 128 \
    --ifs-rm-bs 64 \
    --finetune-steps 1000 \
    --rm-idx-path ./data/rm-idx/fmt/6k/$i/rm-idx.pkl \
    --resume-path $root/train/sgld/0k/$i/train-fin-model.pkl \
    --save-dir    $root/remove/sgld/finetune/6k/$i \
    --save-name finetune
done
