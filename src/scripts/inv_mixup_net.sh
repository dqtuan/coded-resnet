
name="vanilla"

#================= mnist =================
dataset="mnist"
# train
# python ./train_mixup_inv_net.py --name $name --mixup_hidden --mixup_alpha 2 --optimizer sgd --epochs 200 --resume 201 --resume_g 100  --lr 0.001 --dataset mnist --nactors 4 --init_ds 2 --evalFnet --concat_input

# python ./train_mixup_inv_net_reg.py --name $name --epochs 50 --resume 0 --lr 0.0001 --dataset $dataset --lamb 100 --init_ds 2 --mixup_hidden --mixup_alpha 2

# python ./train_mixup_inv_net_freezing.py --name mixup_hidden --resume 201 --resume_g 51  --dataset mnist --nactors 2 --init_ds 2

# python ./train_mixup_inv_net.py --name mixup_hidden --resume 201 --resume_g 100 --dataset mnist --nactors 2 --init_ds 2 --evalFgan

# eval
# python ./train_mixup_inv_net.py --name mixup_hidden --resume 201 --resume_g 100  --dataset mnist --nactors 2 --evalFgan --init_ds 2

# python ./train_mixup_inv_net.py --name mixup_hidden --resume 201 --resume_g 191  --dataset mnist --nactors 4 --evalFgan --init_ds 2 #--concat_input # old

# python ./train_mixup_inv_net.py --name mixup_hidden --resume 201 --resume_g 191  --dataset mnist --nactors 4 --plot_fused --init_ds 2 #--concat_input # old

# python ./train_mixup_inv_net_reg.py --name $name --resume 50 --dataset $dataset --init_ds 2 --plot_fused --nactors 4

# generate images
# python ./train_mixup_inv_net.py --name mixup_hidden --resume 201 --dataset mnist --nactors 4 --init_ds 2 --sampleImg --batch_size 512 --reduction mean


#================= fashion =================
# train
# python ./train_mixup_inv_net_freezing.py --name mixup_hidden --resume 91 --resume_g 200  --dataset fashion --nactors 2 --init_ds 2
dataset="fashion"
# python ./train_mixup_inv_net_reg.py --name $name --epochs 50 --resume 50 --lr 0.0001 --dataset $dataset --lamb 100 --init_ds 2 --mixup_hidden --mixup_alpha 2

# eval
# python ./train_mixup_inv_net.py --name mixup_hidden --resume 91 --resume_g 200  --dataset fashion --nactors 2 --evalInv --init_ds 2

# python ./train_mixup_inv_net.py --name mixup_hidden --resume 91 --resume_g 100  --dataset fashion --nactors 2 --evalFnet --init_ds 2

# python ./train_mixup_inv_net.py --dataset fashion --name mixup_hidden --nactors 2 --resume 91 --resume_g 200 --init_ds 2 --evalFgan

# generate images
# python ./train_mixup_inv_net.py --name mixup_hidden --mixup_hidden --mixup_alpha 2 --dataset fashion --nactors 2 --init_ds 2 --resume 91 --sampleImg --batch_size 512 --reduction sum

# python ./train_mixup_inv_net.py --name mixup_hidden --mixup_hidden --mixup_alpha 2 --dataset fashion --nactors 4 --init_ds 2 --resume 91 --sampleImg --batch_size 512 --reduction mean

# python ./train_mixup_inv_net.py --name mixup_hidden --dataset fashion --nactors 2 --resume 91 --resume_g 150 --evalFgan --init_ds 2 --batch_size 512

#================= cifar10 =================
dataset='cifar10'
# train
# python ./train_mixup_inv_net.py --name mixup --dataset cifar10 --resume 0 --lr 0.1 --epochs 200 --n_epochs 50 --n_epochs_decay 150 --mixup_hidden --mixup_alpha 2 --lr_policy linear
# python ./train_mixup_inv_net.py --name mixup --dataset cifar10 --resume 0 --lr 0.1 --epochs 200 --n_epochs 50 --n_epochs_decay 150 --mixup_hidden --mixup_alpha 2 --lr_policy linear
# python ./train_mixup_inv_net_reg.py --name vanilla --epochs 50 --resume 0 --lr 0.0001 --dataset $dataset --lamb 100 --init_ds 1

# eval
# python ./train_mixup_inv_net.py --name mixup_hidden --dataset cifar10 --nactors 2 --resume 1001 --resume_g 191 --evalInv --init_ds 2
# python ./train_mixup_inv_net.py --name vanilla1 --dataset cifar10 --nactors 2 --resume 191 --resume_g 191 --evalInv


# generate images
# python ./train_mixup_inv_net.py --name mixup_hidden1 --mixup_hidden --mixup_alpha 2 --resume 301 --sampleImg --nactors 2
python ./train_mixup_inv_net.py --name mixup --dataset cifar10 --resume 200 --nactors 2 --sampleImg --batch_size 512
