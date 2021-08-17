#!/usr/bin/env bash

# Evaluate the best validation model on Scene Flow test set
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py \
--mode test \
--data_dir data/apolloscape \
--dataset_name apolloscape \
--checkpoint_dir checkpoints/apolloscape_2nd \
--batch_size 4 \
--val_batch_size 1 \
--img_height 384 \
--img_width 1248 \
--val_img_height 384 \
--val_img_width 1248 \
--feature_type ganet \
--feature_pyramid \
--refinement_type hourglass \
--milestones 20,30,40,50,60 \
--max_epoch 100 \
--evaluate_only

# Evaluate a specific model on Scene Flow test set
CUDA_VISIBLE_DEVICES=0 python train.py \
--mode test \
--data_dir data/apolloscape \
--dataset_name apolloscape \
--checkpoint_dir checkpoints/apolloscape_2nd \
--pretrained_aanet checkpoints/apolloscape_2nd/aanet_best.pth \
--batch_size 4 \
--val_batch_size 1 \
--img_height 384 \
--img_width 1248 \
--val_img_height 384 \
--val_img_width 1248 \
--feature_type ganet \
--feature_pyramid \
--refinement_type hourglass \
--milestones 20,30,40,50,60 \
--max_epoch 100 \
--evaluate_only
