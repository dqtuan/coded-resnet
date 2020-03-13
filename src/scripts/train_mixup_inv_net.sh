# mixup-hidden
# python ./train_mixup_inv_net.py --model_name mixup_hidden --mixup_hidden --mixup_alpha 2 --optimizer sgd --epochs 200 --resume 601 --lr 0.0005

# generate data for fusion
# python ./train_mixup_inv_net.py --model_name mixup_hidden --mixup_hidden --mixup_alpha 2 --resume 601 --generate_fused_image
# rm ../data/pix2pix/data.npy

# evaluate fusion net mixup-hidden
python ./train_mixup_inv_net.py --model_name mixup_hidden --mixup_hidden --mixup_alpha 2 --resume 601 --eval_fusion_net

# adversarial
# python ./demo_aai_net.py --model_name aai --resume 200
