
###================= mnist =================
iname="mixup"
fname="distill"
### MNIST ###
# dataset="mnist"
# init=2
# nactors=4
# resume=201
# resume_g=101

####================= Fashion ###=================
# dataset="fashion"
# init=2
# nactors=4
# resume=121
# resume_g=21
#
####================= Cifar10 ###=================
dataset="cifar10"
init=1
nactors=4
resume=200
resume_g=21 #501

python ./train_mixup_inv_net.py --dataset $dataset --iname $iname --fname $fname --resume $resume --resume_g $resume_g  --nactors $nactors --init_ds $init --evalDistill
