!/usr/bin/env bash

# Train on Apolloscape training set
CUDA_VISIBLE_DEVICES=2,3 python train.py \
--mode val \
--data_dir data/apolloscape \
--dataset_name apolloscape \
--checkpoint_dir checkpoints/apolloscape_large_v2_1st \
--pretrained_aanet pretrained/aanet+_sceneflow-d3e13ef0.pth \
--batch_size 2 \
--val_batch_size 1 \
--img_height 672 \
--img_width 1440 \
--val_img_height 672 \
--val_img_width 1440 \
--max_disp 192 \
--feature_type ganet \
--feature_pyramid \
--refinement_type hourglass \
--milestones 15,30,50,70,90 \
--max_epoch 100 \
--save_ckpt_freq 10 
# --no_validate