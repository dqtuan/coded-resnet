#!/bin/bash
# python -u ./train_resnet.py  --name resnet --resume 0 --dataset fashion --input_nc 1 --epochs 200 --lr 0.01 --evalDistill

# python ./train_integrated_resnet.py --name mixup_resnet --epochs 400 --input_nc 3 --output_nc 3 --dataset cifar10 --nactors 2 --batch_size 128 --lr 0.01 --resume 291 --resume_g 400

#MNIST
# python -u ./train_resnet.py --name resnet --dataset mnist --input_nc 1 --epochs 200 --lr 0.01 --resume 21 --resume_g 291 --evalDistill


# CIAR10
#
# python -u ./train_mixup_resnet.py  --name mixup_resnet --dataset cifar10 --input_nc 3 --epochs 400 --lr 0.01 --resume 21 --resume_g 291 --evalDistill

# python -u ./train_mixup_resnet.py  --name mixup_resnet --mixup_hidden --mixup_alpha 2 --dataset cifar10 --input_nc 3 --epochs 300 --lr 0.01 --resume 0

# python ./train_integrated_resnet.py --name mixup_resnet --epochs 400 --input_nc 3 --output_nc 3 --dataset cifar10 --nactors 2 --batch_size 128 --lr 0.01 --resume 291 --resume_g 400
