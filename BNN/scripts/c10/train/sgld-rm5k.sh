cd $1

root='./exp_data/c10-smcnn'

for i in {1..5}
do
python train.py \
    --arch small-cnn \
    --prior-sig 0.1 \
    --dataset cifar10 \
    --burn-in-steps 20000 \
    --optim sgld \
    --batch-size 128 \
    --sp-init 0.01 \
    --sp-decay-exp -0.5005 \
    --lr 0.01 \
    --lr-decay-rate 1 \
    --lr-decay-freq 2000 \
    --explore-steps 5000 \
    --eval-freq 2000 \
    --rm-idx-path ./data/rm-idx/c10/5k/$i/rm-idx.pkl \
    --init-model-path $root/premodel/sgld/$i/premodel-fin-model.pkl \
    --save-dir $root/train/sgld/5k/$i \
    --save-name train
done
