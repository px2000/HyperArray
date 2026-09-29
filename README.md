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
├── utils/
│   ├── DataProcess/                 # Data processing utilities
│   ├── getdataset/                  # Dataset loading utilities
│   └── my_utils/                    # General utility functions
│
├── data/
│   ├── train/                       # Training hyperspectral images
│   ├── val/                         # Validation hyperspectral images
│   └── test/                        # Test hyperspectral images
│
├── mask/
│   ├── train/                       # Masks used for training
│   └── test/                        # Calibrated masks used for testing
│
├── model_zoo/
│   └── net.pth                      # Pretrained model checkpoint
│
├── train.py                         # Reconstruction network training
└── train.sh                         # Example training script
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

To improve reconstruction robustness to mask variations among different camera modules, calibrated masks are divided into local patches and used to train a denoising diffusion probabilistic model.

The diffusion model learns the distribution of mask variations across different camera modules.

After training, new mask patches are generated from Gaussian noise and reassembled into full-resolution masks.

Example:

```bash
python diffusion/train_diffusion.py \
    --mask_dir ./Data/masks/train \
    --image_size 128 \
    --batch_size 8
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
    --mask_dir ./Data/masks/train \
    --train_data_path ./Data/train/ \
    --val_data_path ./Data/val/ \
    --model_dir ./model_SRNet \
```

---

# 🔄 Model Checkpoints

To resume training from a checkpoint, add the `--resume` option in `train.sh`:

```bash
python train.py \
    --resume ./model_SRNet/checkpoint.pth
```

To evaluate a pretrained model, specify the model path using --pretrained_model_path in test.sh:

```bash
python train.py \
    --pretrained_model_path ./model_zoo/net.pth
```

---

## 📬 Contact

- For questions, please contact: [bian@bit.edu.cn](mailto:bian@bit.edu.cn) or open an issue on this GitHub repository.
