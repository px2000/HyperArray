import argparse
import os

import h5py
import hdf5storage
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from architecture import model_generator
from utils.getdataset import TestDataset_MOS


parser = argparse.ArgumentParser(description="Reconstruct hyperspectral images using a single calibrated mask")

parser.add_argument("--gpu_id", type=str, default="0", help="GPU id")
parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
parser.add_argument("--start_dir", type=int, nargs=2, default=(0, 0), help="Top-left crop position: H W")
parser.add_argument("--image_size", type=int, nargs=2, default=(2048, 2448), help="Input image size: H W")

parser.add_argument("--method", type=str, default="UM_SRNet", help="Model name")
parser.add_argument("--mask_path", type=str, required=True, help="Path to a single calibrated mask (.mat)")
parser.add_argument("--pretrained_model_path", type=str, required=True, help="Path to pretrained model")

parser.add_argument("--image_folder", type=str, required=True, help="Folder containing measurement images")
parser.add_argument("--save_folder", type=str, default="./results/", help="Output folder")

opt = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

_mask_cache = {}

def load_mask(mask_path):

    if mask_path not in _mask_cache:
        mask_init = hdf5storage.loadmat(mask_path)['mask']
        mask = np.maximum(mask_init, 0)
        mask = mask / mask.max()
        mask = mask.astype(np.float32)
        _mask_cache[mask_path] = torch.from_numpy(mask)

    mask_cpu = _mask_cache[mask_path]
    mask_patch = mask_cpu[:, opt.start_dir[0]: opt.start_dir[0] + opt.image_size[0],
                          opt.start_dir[1]: opt.start_dir[1] + opt.image_size[1]]
    mask_patch = mask_patch.cuda(non_blocking=True).unsqueeze(0)
    return mask_patch


def main():
    cudnn.benchmark = True

    model = model_generator(opt.method, opt.pretrained_model_path)

    if torch.cuda.is_available():
        model = model.cuda()

    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")

    mask_test = load_mask(opt.mask_path)
    print(f"Using mask: {opt.mask_path}")
    print(f"Mask shape: {tuple(mask_test.shape)}")

    os.makedirs(opt.save_folder, exist_ok=True)

    test_list = sorted(os.listdir(opt.image_folder))

    test_data = TestDataset_MOS(data_path=opt.image_folder, data_list=test_list, start_dir=opt.start_dir, image_size=opt.image_size, arg=False)
    test_loader = DataLoader(dataset=test_data, batch_size=opt.batch_size, shuffle=False, num_workers=0, pin_memory=True, drop_last=False)

    with torch.no_grad():
        for i, (mos, mos_name) in enumerate(test_loader):
            mos = mos.cuda(non_blocking=True)

            current_batch = mos.shape[0]
            mask_batch = mask_test.expand(current_batch, -1, -1, -1)

            outputs = model(mos, mask_batch)
            outputs = outputs.clamp(0, 1)

            for k in range(current_batch):
                name = mos_name[k]

                output_hsi = outputs[k].squeeze().cpu().numpy()
                input_mos = mos[k].squeeze().cpu().numpy()

                base_name = os.path.splitext(name)[0]
                out_path = os.path.join(opt.save_folder, f"HSI_R_{base_name}.h5")

                with h5py.File(out_path, "w") as f:
                    f["mos"] = input_mos
                    f["hsi_R"] = output_hsi

                print(
                    f"[{i:05d}] {name} | "
                    f"mask: {os.path.basename(opt.mask_path)} | "
                    f"saved: {out_path}"
                )


if __name__ == "__main__":
    main()
