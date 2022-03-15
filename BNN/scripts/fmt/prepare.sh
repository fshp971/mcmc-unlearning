cd $1

root='./exp_data/fmt-lenet'

for i in {1..5}
do
python prepare_model.py \
    --arch lenet \
    --pretrain-dataset mnist \
    --finetune-dataset fashion-mnist \
    --optim sgd \
    --momentum 0.9 \
    --batch-size 128 \
    --lr 0.1 \
    --burn-in-steps 2000 \
    --save-freq 10000 \
    --eval-freq 1000 \
    --save-dir $root/premodel/sgld/$i \
    --save-name premodel
done
