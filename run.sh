# Adam
CUDA_VISIBLE_DEVICES=1, python train_quant.py   --dataset 'cifar100' \
                                                --arch 'resnet20_quant' \
                                                --epochs 400 \
                                                --bit_weight 2 \
                                                --bit_act 2 \
                                                --optimizer 'Adam' \
                                                --baseline False \
                                                --weight_decay 1e-4 \
                                                --lr 0.001 \
                                                --initial_tr 5e-3 \
                                                --momentum_tr 0.99 \
                                                --load_pretrain True \
                                                --pretrain_path '../results/ResNet20_CIFAR100/fp/checkpoint/last_checkpoint.pth' \
                                                --log_dir '../results/210915_multi_bit_quant/CIFAR100/W2A2-ours-Adam-tr0.005-M0.99/'

# SGD
CUDA_VISIBLE_DEVICES=1, python train_quant.py   --dataset 'cifar100' \
                                                --arch 'resnet20_quant' \
                                                --epochs 400 \
                                                --bit_weight 2 \
                                                --bit_act 2 \
                                                --optimizer 'SGD' \
                                                --baseline False \
                                                --weight_decay 1e-4 \
                                                --lr 0.1 \
                                                --initial_tr 5e-3 \
                                                --momentum_tr 0.99 \
                                                --load_pretrain True \
                                                --pretrain_path '../results/ResNet20_CIFAR100/fp/checkpoint/last_checkpoint.pth' \
                                                --log_dir '../results/210915_multi_bit_quant/CIFAR100/W2A2-ours-SGD-tr0.005-M0.99/'
