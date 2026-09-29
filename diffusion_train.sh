#!/bin/bash

python diffusion_train.py \
    --device 0 \
    --train_paths \
        ../SRNet_GST/MASK_distance_1/mask_260122_f4_25mm_001_nearest.mat \
        ../SRNet_GST/MASK_distance_1/mask_260122_f4_25mm_005_nearest.mat \
        ../SRNet_GST/MASK_distance_1/mask_260122_f4_25mm_006_nearest.mat \
        ../SRNet_GST/MASK_distance_1/mask_260121_f4_25mm_009_nearest.mat \
        ../SRNet_GST/MASK_distance_2/mask_260121_f4_25mm_012_nearest.mat \
        ../SRNet_GST/MASK_distance_2/mask_260121_f4_25mm_014_99percent999_nearest.mat \
        ../SRNet_GST/MASK_distance_2/mask_260121_f4_25mm_015_nearest.mat \
        ../SRNet_GST/MASK_distance_3/mask_260120_f4_25mm_021_nearest.mat \
        ../SRNet_GST/MASK_distance_3/mask_260120_f4_25mm_023_nearest.mat \
        ../SRNet_GST/MASK_distance_3/mask_260127_f4_25mm_026_nearest.mat \
    --save_mask_dir ./save_model/260428_diffusion_distance_train10_128/mat/ \
    --save_ckpt_dir ./save_model/260428_diffusion_distance_train10_128/pt/ \
