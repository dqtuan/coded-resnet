# python ./train_integrated_net.py --name mixup_hidden --mixup_hidden --mixup_alpha 2 --optimizer sgd --epochs 301 --resume 201 --input_nc 1 --output_nc 1 --dataset mnist --nactors 2 --init_ds 2

# python ./train_integrated_net.py --name mixup_hidden1 --mixup_hidden --mixup_alpha 2 --optimizer sgd --epochs 301 --resume 291 --input_nc 3 --output_nc 3 --dataset cifar10 --nactors 2 --batch_size 128 --init_ds 1 --lr 0.001 --resume_g 580

python ./train_integrated_net.py --name mixup --dataset cifar10 --nactors 4 --resume 200 --resume_g 0 --epochs 200 --lr 0.001 --no_dropout --n_epochs 30 --niter_decay 180 --lambda_L1 10

# python ./train_integrated_net.py --name mixup --dataset cifar10 --nactors 2 --resume 200 --resume_g 121 --epochs 100 --lr 0.0002 --no_dropout --n_epochs 30 --niter_decay 80 --lambda_L1 10


# python ./train_disgan_net.py --name mixup --dataset cifar10 --nactors 2 --resume 200 --resume_g 0 --epochs 200 --lr 0.01  --lambda_distill 20 --lambda_L1 10
