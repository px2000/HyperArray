import argparse
import os

import hdf5storage
import numpy as np
import torch
from scipy.signal import savgol_filter
from torch.utils.data import DataLoader, IterableDataset

from ddpm.unet import UNet
from ddpm.diffusion import GaussianDiffusion


def parse_args():
    parser = argparse.ArgumentParser(description="Train diffusion model and generate mask.")

    parser.add_argument("--device", type=str, default="0", help="CUDA device ID")
    parser.add_argument("--train_paths", type=str, nargs="+", required=True, help="Paths to calibrated training mask .mat files")
    parser.add_argument("--save_mask_dir", type=str, default="./save_model/diffusion/mat/", help="Directory for generated full masks")
    parser.add_argument("--save_ckpt_dir", type=str, default="./save_model/diffusion/pt/", help="Directory for diffusion checkpoints")
    parser.add_argument("--patch", type=int, default=128, help="Training patch size")
    parser.add_argument("--steps_per_epoch", type=int, default=1000, help="Number of sampled patches per epoch")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path for resuming training")
    parser.add_argument("--timesteps", type=int, default=1000, help="Number of diffusion timesteps")
    parser.add_argument("--inner_channel", type=int, default=64, help="UNet base channel dimension")
    parser.add_argument("--out_h", type=int, default=2048, help="Output mask height")
    parser.add_argument("--out_w", type=int, default=2448, help="Output mask width")
    parser.add_argument("--sample_batch_size", type=int, default=4, help="Batch size used for diffusion sampling")
    parser.add_argument("--flag_filter", action="store_true", help="Apply Savitzky-Golay filtering to generated mask patches")

    return parser.parse_args()


def load_one_mask_mat(mask_path: str) -> torch.Tensor:
    """Load one calibrated mask as a [61, H, W] float32 tensor."""
    mask_init = hdf5storage.loadmat(mask_path)["mask"]
    mask = np.maximum(mask_init, 0)
    return torch.from_numpy(mask).float()


class RandomMaskPatchDataset(IterableDataset):
    """Randomly crop [61, patch, patch] patches from calibrated masks."""

    def __init__(self, mask_paths, patch=128, steps_per_epoch=1000, seed=0):
        super().__init__()
        self.patch = patch
        self.steps_per_epoch = steps_per_epoch
        self.seed = seed
        self.masks = [load_one_mask_mat(p) for p in mask_paths]

        c, h, w = self.masks[0].shape
        assert c == 61, f"Expected 61 channels, got {c}"
        assert h >= patch and w >= patch, f"Mask is too small for patch={patch}"

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = 0 if worker_info is None else worker_info.id
        rng = np.random.default_rng(self.seed + worker_id)

        for _ in range(self.steps_per_epoch):
            mask_idx = int(rng.integers(0, len(self.masks)))
            mask = self.masks[mask_idx]

            _, h, w = mask.shape
            y = int(rng.integers(0, h - self.patch + 1))
            x = int(rng.integers(0, w - self.patch + 1))

            yield mask[:, y:y + self.patch, x:x + self.patch]


def stitch_mask(patches, out_h=2048, out_w=2448, patch_size=128):
    """Stitch generated patches into one full-resolution mask."""
    n, c, ph, pw = patches.shape
    assert c == 61
    assert ph == patch_size and pw == patch_size

    n_rows = int(np.ceil(out_h / patch_size))
    n_cols = int(np.ceil(out_w / patch_size))
    need = n_rows * n_cols
    assert n >= need, f"Not enough patches: need {need}, got {n}"

    full_h = n_rows * patch_size
    full_w = n_cols * patch_size
    full_mask = np.zeros((c, full_h, full_w), dtype=patches.dtype)

    idx = 0
    for r in range(n_rows):
        for col in range(n_cols):
            y0 = r * patch_size
            x0 = col * patch_size
            full_mask[:, y0:y0 + patch_size, x0:x0 + patch_size] = patches[idx]
            idx += 1

    return full_mask[:, :out_h, :out_w]


@torch.no_grad()
def generate_full_mask(
    diffusion,
    save_path,
    out_h=2048,
    out_w=2448,
    patch=128,
    batch_each=4,
    flag_filter=True,
):
    diffusion.eval()

    n_rows = int(np.ceil(out_h / patch))
    n_cols = int(np.ceil(out_w / patch))
    need = n_rows * n_cols
    buffer = []

    while len(buffer) < need:
        samples = diffusion.sample(batch_size=batch_each)
        samples = samples.detach().cpu().numpy()

        for k in range(samples.shape[0]):
            data = samples[k].astype(np.float32)

            if flag_filter:
                c, h, w = data.shape
                spectra = data.reshape(c, h * w).T
                spectra = savgol_filter(
                    spectra,
                    window_length=11,
                    polyorder=3,
                    axis=-1,
                    mode="nearest",
                )
                data = spectra.T.reshape(c, h, w)
                data = np.maximum(data, 0)

            buffer.append(data)

            if len(buffer) >= need:
                break

        del samples
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"[Generate] {len(buffer)}/{need} patches generated.")

    patches = np.stack(buffer[:need], axis=0)
    full_mask = stitch_mask(
        patches,
        out_h=out_h,
        out_w=out_w,
        patch_size=patch,
    )

    bands = np.linspace(400, 1000, 61, dtype=np.int32)
    hdf5storage.savemat(
        save_path,
        {"bands": bands, "mask": full_mask},
    )

    print(f"[Generate] Full mask saved to: {save_path}")


def train_diffusion(args):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    os.makedirs(args.save_mask_dir, exist_ok=True)
    os.makedirs(args.save_ckpt_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = RandomMaskPatchDataset(
        args.train_paths,
        patch=args.patch,
        steps_per_epoch=args.steps_per_epoch,
        seed=123,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        num_workers=4,
        pin_memory=True,
    )

    model = UNet(
        in_channel=61,
        out_channel=61,
        inner_channel=args.inner_channel,
    ).to(device)

    diffusion = GaussianDiffusion(
        model=model,
        image_size=args.patch,
        channels=61,
        timesteps=args.timesteps,
    ).to(device)

    optimizer = torch.optim.AdamW(
        diffusion.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
    )

    start_epoch = 1
    global_step = 0

    if args.resume is not None and os.path.isfile(args.resume):
        print(f"[Resume] Loading checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location="cpu")

        if "model" not in ckpt:
            raise KeyError("Checkpoint has no 'model' key.")

        model.load_state_dict(ckpt["model"], strict=True)

        if "opt" in ckpt:
            optimizer.load_state_dict(ckpt["opt"])
        if "epoch" in ckpt:
            start_epoch = int(ckpt["epoch"]) + 1
        if "global_step" in ckpt:
            global_step = int(ckpt["global_step"])

        print(
            f"[Resume] start_epoch={start_epoch}, "
            f"global_step={global_step}"
        )

    diffusion.train()

    for epoch in range(start_epoch, args.epochs + 1):
        for step, x in enumerate(train_loader, 1):
            x = x.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            loss = diffusion(x)
            loss.backward()
            optimizer.step()

            global_step += 1

            if step % 50 == 0:
                print(
                    f"[E{epoch:03d}]"
                    f"[{step:04d}/{args.steps_per_epoch}] "
                    f"loss={loss.item():.6f}"
                )

        ckpt_path = os.path.join(
            args.save_ckpt_dir,
            f"net_{epoch:03d}epoch.pth",
        )

        torch.save(
            {
                "epoch": epoch,
                "global_step": global_step,
                "model": model.state_dict(),
                "opt": optimizer.state_dict(),
            },
            ckpt_path,
        )

        print(f"[Checkpoint] Saved: {ckpt_path}")

    final_mask_path = os.path.join(
        args.save_mask_dir,
        f"full_mask_{args.epochs:03d}.mat",
    )

    generate_full_mask(
        diffusion=diffusion,
        save_path=final_mask_path,
        out_h=args.out_h,
        out_w=args.out_w,
        patch=args.patch,
        batch_each=args.sample_batch_size,
        flag_filter=args.flag_filter,
    )


if __name__ == "__main__":
    args = parse_args()
    train_diffusion(args)
