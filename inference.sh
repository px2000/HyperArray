#!/bin/bash

python inference.py \
    --gpu_id 0 \
    --method UM_SRNet \
    --mask_path ./mask/mask.mat \
    --pretrained_model_path ./model_zoo/net.pth \
    --image_folder ./data/measurement/ \
    --save_folder ./results/
