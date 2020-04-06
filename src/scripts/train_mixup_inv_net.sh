

# generate data for fusion
# python ./train_mixup_inv_net.py --model_name mixup_hidden --mixup_hidden --mixup_alpha 2 --resume 1001 --generate_fused_image
# rm ../data/pix2pix/data.npy

# mixup-hidden
# python ./train_mixup_inv_net.py --model_name mixup_hidden --mixup_hidden --mixup_alpha 2 --optimizer sgd --epochs 200 --resume 801 --lr 0.01


# evaluate fusion net mixup-hidden
# python ./train_mixup_inv_net.py --model_name mixup_hidden --mixup_hidden --mixup_alpha 2 --resume 801 --eval_fusion_net

# adversarial
# python ./demo_aai_net.py --model_name aai --resume 200

# vanilla iResNet
# python ./train_mixup_inv_net.py --model_name vanilla --optimizer sgd --epochs 200 --resume 101 --dataset mnist --eval_invertibility


# generate data for fusion
# rm ../data/pix2pix/data.npy
# python ./train_mixup_inv_net.py --model_name vanilla --resume 101 --generate_fused_image --dataset mnist --nactors 2

# evaluate fusion net mixup-hidden
# python ./train_mixup_inv_net.py --model_name vanilla --resume 101 --eval_invertibility --dataset mnist --nactors 4
# python ./train_mixup_inv_net.py --model_name vanilla --resume 101 --eval_fusion_net --dataset mnist --nactors 4

# vanilla iResNet
python ./train_mixup_inv_net.py --model_name mixup_hidden --mixup_hidden --mixup_alpha 2 --optimizer sgd --epochs 201 --resume 201 --dataset mnist --nactors 4 --eval_fusion_net

python ./train_mixup_inv_net.py --model_name mixup_hidden --mixup_hidden --mixup_alpha 2 --optimizer sgd --epochs 201 --resume 201 --dataset mnist --nactors 4 --eval_fusion_net --concat_input

# python ./train_mixup_inv_net.py --model_name mixup_input --mixup --optimizer sgd --epochs 200 --resume 1 --dataset mnist

# generate data for fusion
# python ./train_mixup_inv_net.py --model_name mixup_hidden --mixup_hidden --mixup_alpha 2 --resume 201 --generate_fused_image --dataset mnist
# rm ../data/pix2pix/data.npy

# python ./train_fusion_net.py --lr 0.001 --lr_policy step --model_name mixup_hidden_fusion --input_nc 1 --output_nc 1 --dataset mnist

# python ./train_mixup_inv_net.py --model_name mixup_hidden --mixup_hidden --mixup_alpha 2 --resume 201 --eval_fusion_net --dataset mnist
