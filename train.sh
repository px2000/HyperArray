#!/bin/bash

python train.py --device 0 \
                          --method  UM_SRNet \
                          --mask_dir  /mask/ \
                          --train_data_path /data/Train/ \
                          --val_data_path /data/Val/ \
                          --model_dir  ./save_model/... 
                        

