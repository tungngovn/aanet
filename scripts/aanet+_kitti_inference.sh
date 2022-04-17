#!/usr/bin/env bash

# Inference on KITTI 2012 test set for submission
CUDA_VISIBLE_DEVICES=0 python inference.py \
--mode test \
--data_dir data/KITTI/kitti_2012/data_stereo_flow \
--dataset_name KITTI2012 \
--pretrained_aanet pretrained/aanet+_kitti12-56157808.pth \
--batch_size 1 \
--img_height 384 \
--img_width 1248 \
--feature_type ganet \
--feature_pyramid \
--refinement_type hourglass \
--no_intermediate_supervision \
--output_dir output/kitti12_test
