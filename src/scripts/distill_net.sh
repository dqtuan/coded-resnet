iname="mixup"
fname="distill"

# if ; then
# elif ; then
# else
# fi
# tensorboard --logdir=runs
###================= mnist =================

#### MNIST ###
# dataset="mnist"
# init=2
# nactors=4
# resume=201
# resume_g=241
# lr=0.00001
# epochs=100
# n_epochs=20
# niter_decay=80
# lambda_L1=0
# lambda_distill=10
#36690
####================= Fashion ###=================
dataset="fashion"
init=2
nactors=4
resume=121
resume_g=191
epochs=100
lr=0.0002
lambda_L1=1
lambda_distill=100

# nohup sh scripts/distill_net.sh  > ../results/logs/fnet_distill_fashion_mixup_4.txt &
# alias tf='tail -n 100 ../results/logs/fnet_distill_fashion_mixup_4.txt'
#36217
####================= Cifar10 ###=================
# dataset="cifar10"
# init=1
# nactors=4
# resume=200
# resume_g=140
# lr=0.0001
# epochs=200
# lambda_L1=1
# lambda_distill=100
#11098
# nohup sh scripts/distill_net.sh  > ../results/logs/fnet_distill_cifar10_mixup_4.txt &
# alias tf='tail -n 100 ../results/logs/fnet_distill_cifar10_mixup_4.txt'

# rm -r ../results/runs
python ./train_integrated_net.py --dataset $dataset --fname $fname --iname $iname --nactors $nactors --resume $resume --resume_g $resume_g --init_ds $init --epochs $epochs --lr $lr --batch_size 512 --lambda_L1 $lambda_L1 --lambda_distill $lambda_distill
