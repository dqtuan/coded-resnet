python ./CIFAR_inverse.py --nBlocks 7 7 7 --nStrides 1 2 2 --nChannels 32 64 128 --coeff 0.9 --batch 256 --dataset cifar10 --init_ds 1 --inj_pad 13 --powerIterSpectralNorm 20 --nonlin elu --optimizer sgd --vis_server localhost  --vis_port  8097 --save_dir ../results/inverse  --resume 7
# python -m visdom.server or visom in tmux
