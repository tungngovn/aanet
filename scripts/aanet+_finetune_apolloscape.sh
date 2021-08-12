#!/usr/bin/env bash

# Train on Apolloscape training set
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py \
--data_dir data/apolloscape \
--dataset_name apolloscape \
--checkpoint_dir checkpoints/apolloscape \
--pretrained_aanet pretrained/aanet+_sceneflow-d3e13ef0.pth \
--batch_size 6 \
--val_batch_size 6 \
--img_height 960 \
--img_width 3130 \
--val_img_height 960 \
--val_img_width 3130 \
--max_disp 192 \
--feature_type ganet \
--feature_pyramid \
--refinement_type hourglass \
--milestones 40,60,80,90 \
--max_epoch 100 \
--save_ckpt_freq 10 
# --no_validate

