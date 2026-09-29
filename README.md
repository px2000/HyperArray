# A computational camera array for large-scale snapshot hyperspectral imaging（HyperArray）

Lianjie Li\*, Xuan Peng\*, Jingyi Wang, Jianqi Zi, Xilong Dai, Liheng Bian†  (\* Equal contributions, † Corresponding author)

This is the official implementation of "A computational camera array for large-scale snapshot hyperspectral imaging". 
HyperArray reconstructs 61-band hyperspectral datacubes from coded 2D measurements acquired by a computational camera array using UM-SRNet, enabling unified hyperspectral reconstruction across multiple camera modules. 
The repository includes the source code and supporting resources for reproducing the proposed method.

## 📁 Repository Structure

```text
UM-SRNet/
├── architecture/
│   ├── __init__.py                  # Model initialization
│   └── UM_SRNet.py                  # UM-SRNet architecture
│
├── ddpm/
│   ├── diffusion.py                 # Gaussian diffusion process
│   └── unet.py                      # Diffusion UNet architecture
│
├── utils/
│   ├── DataProcess.py               # Data processing utilities
│   ├── getdataset.py                # Dataset loading utilities
│   └── my_utils.py                  # General utility functions
│
├── data/                            # Hyperspectral dataset and measurement
├── mask/                            # Spectral encoding masks
├── model_zoo/                       # Pretrained model checkpoint
│
├── diffusion_train.py               # Diffusion model training
├── diffusion_train.sh               # Example diffusion training script
├── train.py                         # Reconstruction network training
├── train.sh                         # Example training script
├── inference.py                     # Hyperspectral reconstruction
└── inference.sh                     # Example inference script
```

> The directory structure can be adjusted according to your local dataset paths.

---

# 🚀 Quick Start

## Environment

The code was tested under the following environment:

- **Operating System:** Ubuntu 20.04
- **GPU:** NVIDIA GeForce RTX 4090
- **Python:** 3.11.13
- **PyTorch:** 2.5.1
- **Torchvision:** 0.20.1
- **CUDA:** 12.1

---

# 🌫️ Diffusion-based Mask Generation

Training masks are MAT files containing a `mask` array in channel-first form (`C × H × W`). Place the calibrated masks in the `mask/` directory and specify their paths in `diffusion_train.sh`.  
The paths and training parameters can be configured in `diffusion_train.sh`.

Example:

```bash
python diffusion_train.py \
    --device 0 \
    --train_paths \
        /mask/... \
    --save_mask_dir ./save_model/mat/ \
    --save_ckpt_dir ./save_model/pt/ 
```

The generated masks can then be used for multi-mask reconstruction training.

---

# 🏋️ Training

Training samples are HDF5 files containing an `hsi` dataset in channel-first form (`C × H × W`). Place the training and validation files in `data/train/` and `data/val/`, respectively, and place the training masks in the corresponding `mask/` directory. 
The paths and training parameters can be configured in `train.sh`.

Example:

```bash
python train.py \
    --device 0 \
    --method UM-SRNet \
    --mask_dir ./data/masks/ \
    --train_data_path ./data/train/ \
    --val_data_path ./data/val/ \
    --model_dir ./model_SRNet
```

---

# 🔄 Model Checkpoints

To resume training from a checkpoint, add the `--resume` option in `train.sh`:

```bash
python train.py \
    --resume ./model_SRNet/checkpoint.pth
```

To evaluate a pretrained model, specify the model path using `--pretrained_model_path` in `test.sh`:

```bash
python train.py \
    --pretrained_model_path ./model_zoo/net.pth
```

---

# 🔍 Inference

To reconstruct hyperspectral images from measurements, configure the mask path, pretrained model, input folder, and output folder in `inference.sh`:

```bash
sh inference.sh
    --mask_path ./mask/mask.mat \
    --pretrained_model_path ./model_zoo/net.pth \
    --image_folder ./data/measurement/ \
    --save_folder ./results/
```

---

## 📬 Contact

- For questions, please contact [bian@bit.edu.cn](mailto:bian@bit.edu.cn) or [lianjie_li@bit.edu.cn](mailto:lianjie_li@bit.edu.cn), or open an issue in this GitHub repository.
