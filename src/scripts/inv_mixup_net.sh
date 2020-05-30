
###================= mnist =================
iname="mixup"
fname="disgan"
#### MNIST ###
# dataset="mnist"
# init=2
# nactors=4
# resume=201
# resume_g=100

####================= Fashion ###=================
# dataset="fashion"
# init=2
# nactors=2
# resume=121
# resume_g=191
#
####================= Cifar10 ###=================
dataset="cifar10"
init=1
nactors=2
resume=200
resume_g=71 #501

python ./train_mixup_inv_net.py --dataset $dataset --iname $iname --fname $fname --resume $resume --resume_g $resume_g  --nactors $nactors --init_ds $init --evalFgan
