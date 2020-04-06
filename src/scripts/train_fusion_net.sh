# python ./train_fusion_net.py --lr 0.001 --lr_policy step --model_name mixup_fusion_800

# mnist
python ./train_fusion_net.py --lr 0.01 --lr_policy step --model_name mixup_hidden --input_nc 1 --output_nc 1 --dataset mnist --nactors 4 --niter_decay 50 --niter 50 --concat_input

python ./train_fusion_net.py --lr 0.01 --lr_policy step --model_name mixup_hidden --input_nc 1 --output_nc 1 --dataset mnist --nactors 4 --niter_decay 50 --niter 50
