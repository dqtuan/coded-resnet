python ./train_integrated_net.py --name mixup_hidden --mixup_hidden --mixup_alpha 2 --optimizer sgd --epochs 301 --resume 201 --input_nc 1 --output_nc 1 --dataset mnist --nactors 2 --init_ds 2

# python ./train_integrated_net.py --name mixup_hidden1 --mixup_hidden --mixup_alpha 2 --optimizer sgd --epochs 301 --resume 291 --input_nc 3 --output_nc 3 --dataset cifar10 --nactors 2 --batch_size 128 --init_ds 1 --lr 0.001 --resume_g 580

# python ./train_integrated_net.py --name mixup_hidden --mixup_hidden --mixup_alpha 2 --optimizer sgd --epochs 400 --resume 121 --input_nc 1 --output_nc 1 --dataset fashion --nactors 4 --init_ds 2 --lr 0.001
