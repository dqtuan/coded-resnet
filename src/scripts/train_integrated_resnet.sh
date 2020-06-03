# python ./train_integrated_resnet.py --name resnet --epochs 101 --resume 21 --resume_g 0 --input_nc 1 --output_nc 1 --dataset mnist --nactors 4

# python ./train_integrated_resnet.py --name resnet --epochs 301 --input_nc 3 --output_nc 3 --dataset cifar10 --nactors 2 --batch_size 128 --lr 0.01 --resume_g 0

# python ./train_integrated_resnet.py --name resnet --dataset fashion --input_nc 1 --output_nc 1 --resume 31 --resume_g 0 --nactors 4 --epochs 300 --lr 0.01

# python -u ./train_integrated_resnet.py --name resnet --dataset cifar10 --resume 200 --resume_g 290 --evalDistill

# python -u ./train_integrated_resnet.py --name resnet --dataset mnist --input_nc 1 --output_nc 1 --nactors 4 --resume 21  --resume_g 80 --evalDistill
