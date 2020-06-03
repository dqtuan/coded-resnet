#!/bin/bash
#================================= GAN =================================
#================= mnist =================
name="large"

dataset="mnist"
resume=201
nactors=2
reduction="mean"
### train #####
# python ./train_fusion_gan_net.py --name $name --dataset $dataset --nactors $nactors --resume_g 0 --lr 0.0002 --epochs 200 --n_epochs 30 --n_epochs_decay 80 --lr_policy linear --lamb 100 --reduction $reduction

# python ./train_fnet_unet.py --name $name --dataset $dataset --nactors $nactors --resume_g 0 --lr 0.0005 --epochs 100 --n_epochs 30 --n_epochs_decay 80 --lr_policy linear --reduction $reduction

### test #####
resume_g=100
# python ./test_fusion_gan_net.py --name $name --dataset $dataset --nactors $nactors --resume_g $resume_g --resume $resume --init_ds 2 --lamb 100  --batch_size 512 --reduction $reduction --flag_test

# python ./test_fnet_unet.py --name $name --dataset $dataset --nactors $nactors --resume_g $resume_g --resume $resume --init_ds 2 --reduction $reduction --batch_size 512 --flag_test

#================= fashion =================
dataset="fashion"
resume=91
nactors=4
### train #####
# python ./train_fusion_gan_net.py --name $name --dataset $dataset --nactors $nactors --resume_g 0 --lr 0.001 --epochs 100 --n_epochs 30 --n_epochs_decay 80 --lr_policy linear --lamb 100

# python ./train_fnet_unet.py --name $name --dataset $dataset --nactors $nactors --resume_g 0 --lr 0.0005 --epochs 100 --n_epochs 30 --n_epochs_decay 80 --lr_policy linear
# python ./train_fusion_gan_net.py --name mixup_hidden --dataset fashion --nactors 2 --resume_g 101 --lr 0.0001 --n_epochs 100

### test #####
resume_g=100
# python ./test_fusion_gan_net.py --name $name --dataset $dataset --nactors $nactors --resume_g $resume_g --resume $resume --lamb 100 --init_ds 2 --flag_test
## train: GAN: 0.2214 L1 12.7936 Match: 0.8591
#test: GAN: 3.2969 L1 25.5287 Match: 0.7748
#test: GAN: 6.0254 L1 25.6299 Match: 0.7738

# python ./test_fnet_unet.py --name $name --dataset $dataset --nactors $nactors --resume_g $resume_g --resume $resume --init_ds 2 --flag_test
#================= cifar10 =================
dataset='cifar10'
nactors=2
### train #####
# python ./train_fusion_gan_net.py --name mixup --dataset cifar10 --nactors 2 --resume_g 391 --resume 200 --lr 0.00002 --epochs 200 --n_epochs 50 --n_epochs_decay 150 --lr_policy linear --lamb 100 --no_dropout
# python ./train_fusion_gan_net.py --name vanilla --dataset cifar10 --nactors 2 --resume_g 0 --resume 200 --lr 0.0002 --epochs 100 --n_epochs 30 --n_epochs_decay 80 --lr_policy linear --lamb 100
# python ./train_fnet_unet.py --name $name --dataset $dataset --nactors $nactors --resume_g 0 --lr 0.001 --epochs 100 --n_epochs 30 --n_epochs_decay 80 --lr_policy linear --reduction $reduction --no_dropout


### test #####
python ./test_fusion_gan_net.py --name mixup --dataset cifar10 --nactors 2 --resume_g 391 --lamb 100 --resume 200 --no_dropout
# python ./test_fusion_gan_net.py --name vanilla --dataset cifar10 --nactors 4 --resume_g 200 --resume 200 --lamb 100 --batch_size 128
# python ./test_fusion_gan_net.py --name mixup --dataset cifar10 --nactors 2 --resume_g 200 --resume 200 --lamb 100 --batch_size 128
# python ./test_fnet_unet.py --name $name --dataset $dataset --nactors $nactors --resume_g 100 --flag_test --batch_size 128 --no_dropout




# NEGATIVE LR
# 2- cifar10: GAN: 0.2454 L1 10.2802 Match: 0.9795
# test: GAN: 0.4396 L1 17.4630 Match: 0.9491
# Epoch[200](570/586): D-real: 0.1756 D-fake: 0.2584 G-GAN: 0.2728 G-L1 9.7168



#================================= Fnt =================================
#================= mnist =================
### train #####
# python ./train_fusion_net.py --dataset mnist --name mixup_hidden --nactors 2 --resume_g 0 --niter_decay 50 --niter 50  --lr 0.001 --lr_policy step

### test #####

#================= fashion =================
### train #####

### test #####

#================= cifar10 =================
### train #####

### test #####
