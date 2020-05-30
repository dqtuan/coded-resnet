iname="mixup"
fname="distill"
###================= mnist =================

#### MNIST ###
# dataset="mnist"
# init=2
# nactors=4
# resume=201
# resume_g=0

####================= Fashion ###=================
# dataset="fashion"
# init=2
# nactors=4
# resume=121
# resume_g=0
#
####================= Cifar10 ###=================
dataset="cifar10"
init=1
nactors=4
resume=200
resume_g=21 #501

python ./train_integrated_net.py --dataset $dataset --fname $fname --iname $iname --nactors $nactors --resume $resume --resume_g $resume_g --init_ds $init --epochs 200 --lr 0.0005 --no_dropout --n_epochs 30 --niter_decay 180 --lambda_L1 0
