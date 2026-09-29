#!/bin/bash

python diffusion_train.py \
    --device 0 \
    --train_paths \
        /mask/ \
    --save_mask_dir ./save_model/mat/ \
    --save_ckpt_dir ./save_model/pt/ 
