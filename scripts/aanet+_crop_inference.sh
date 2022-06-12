#!/usr/bin/env bash

# Inference
CUDA_VISIBLE_DEVICES=0 python3.6 inference_crop.py \
--mode test \
--data_dir data/apolloscape \
--dataset_name apolloscape \
--pretrained_aanet pretrained/aanet_best.pth \
--batch_size 1 \
--max_disp 192 \
--img_height 960 \
--img_width 3130 \
--feature_type ganet \
--feature_pyramid \
--refinement_type hourglass \
--no_intermediate_supervision \
--output_dir output/apolloscape/offset_images

