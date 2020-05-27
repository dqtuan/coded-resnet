# python ./train_fusion_net.py --lr 0.001 --lr_policy step --model_name mixup_fusion_800

# mnist
# python ./train_fresnet.py --lr 0.001 --lr_policy step --name fresnet --input_nc 1 --output_nc 1 --dataset mnist --nactors 2 --niter_decay 50 --niter 50 --resume_g 0

# python ./train_fresnet.py --lr 0.001 --name fresnet --dataset fashion --nactors 2 --niter_decay 50 --niter 50 --input_nc 1 --output_nc 1 --resume_g 0
python ./test_fresnet.py --name fresnet --dataset mnist --nactors 2 --input_nc 1 --output_nc 1
