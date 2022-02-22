#!/usr/bin/env bash

# Predict
CUDA_VISIBLE_DEVICES=0 python predict.py \
--data_dir demo \
--pretrained_aanet pretrained/aanet+_kitti15-2075aea1.pth \
--img_height 288 \
--img_width 576 \
--feature_type ganet \
--feature_pyramid \
--refinement_type hourglass \
--no_intermediate_supervision
