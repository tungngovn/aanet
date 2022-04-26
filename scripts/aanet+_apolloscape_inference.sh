#!/usr/bin/env bash

# Inference on Apolloscape stereo depth estimation dataset
CUDA_VISIBLE_DEVICES=0 python inference_apollo.py \
--mode test \
--data_dir data/apolloscape \
--dataset_name apolloscape \
--pretrained_aanet checkpoints/apolloscape_large_2nd/aanet_best.pth \
--batch_size 1 \
--max_disp 192 \
--img_height 960 \
--img_width 3072 \
--feature_type ganet \
--feature_pyramid \
--refinement_type hourglass \
--no_intermediate_supervision \
--output_dir output/apolloscape/large_2nd

