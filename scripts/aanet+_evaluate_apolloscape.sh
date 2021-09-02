#!/usr/bin/env bash

# Evaluate the best validation model on Scene Flow test set
CUDA_VISIBLE_DEVICES=0 python train.py \
--mode test \
--data_dir data/apolloscape \
--dataset_name apolloscape \
--checkpoint_dir checkpoints/apolloscape_4th \
--pretrained_aanet checkpoints/apolloscape_2nd/aanet_best.pth \
--batch_size 1 \
--val_batch_size 1 \
--img_height 288 \
--img_width 576 \
--val_img_height 288 \
--val_img_width 576 \
--feature_type ganet \
--feature_pyramid \
--refinement_type hourglass \
--milestones 20,30,40,50,60 \
--max_epoch 100 \
--evaluate_only
