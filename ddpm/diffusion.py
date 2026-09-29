from typing import Dict, List, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ddpm.unet import UNet, UNetOutput


def default(x, d):
    return x if x is not None else d() if callable(d) else d


def normalize_to_neg_one_to_one(img):
    return img * 2 - 1


def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5


def make_beta_schedule(schedule: str, timesteps: int, start=-3, end=3, tau=1):
    if schedule == "linear":
        scale = 1000 / timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)

    if schedule == "cosine":
        import math
        s = 0.008
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps, dtype=torch.float32) / timesteps
        alphas_cumprod = torch.cos((x + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return betas.clamp(1e-6, 0.999)

    if schedule == "sigmoid":
        steps = timesteps + 1
        t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
        v_start = torch.tensor(start / tau).sigmoid()
        v_end = torch.tensor(end / tau).sigmoid()
        alphas_cumprod = (
            -((t * (end - start) + start) / tau).sigmoid() + v_end
        ) / (v_end - v_start)
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0, 0.999).float()

    raise ValueError(f"unsupported schedule: {schedule}")


def extract(a: torch.Tensor, t: torch.Tensor, x_shape: Sequence[int]) -> torch.Tensor:
    b = t.shape[0]
    out = a.gather(0, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        model: UNet,
        image_size: int,
        channels: int = 61,
        timesteps: int = 1000,
        objective: str = "pred_v",
        beta_schedule: str = "sigmoid",
        auto_normalize: bool = True,
    ):
        super().__init__()
        assert objective in {"pred_noise", "pred_x0", "pred_v"}

        self.model = model
        self.image_size = image_size
        self.channels = channels
        self.num_timesteps = timesteps
        self.objective = objective

        betas = make_beta_schedule(beta_schedule, timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]], dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1))

        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer(
            "posterior_log_variance_clipped",
            torch.log(posterior_variance.clamp(min=1e-20)),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

        snr = alphas_cumprod / (1 - alphas_cumprod)

        maybe_clipped_snr = snr.clone()

        if objective == 'pred_noise':
            self.register_buffer('loss_weight', maybe_clipped_snr / snr)
        elif objective == 'pred_x0':
            self.register_buffer('loss_weight', maybe_clipped_snr)
        elif objective == 'pred_v':
            self.register_buffer('loss_weight', maybe_clipped_snr / (snr + 1))

        self.normalize = normalize_to_neg_one_to_one if auto_normalize else (lambda x: x)
        self.unnormalize = unnormalize_to_zero_to_one if auto_normalize else (lambda x: x)

    def predict_start_from_noise(self, x_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_v(self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise
            - extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t: torch.Tensor, t: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def q_posterior(self, x_start: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor):
        model_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_log_variance = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return model_mean, posterior_log_variance

    def model_pred(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        out = self.model(x, t)
        if isinstance(out, UNetOutput):
            out = out.pred
        return out

    def p_mean_variance(self, x: torch.Tensor, t: torch.Tensor, clip_denoised: bool = True):
        model_output = self.model_pred(x, t)

        if self.objective == "pred_noise":
            x_recon = self.predict_start_from_noise(x, t, model_output)
        elif self.objective == "pred_x0":
            x_recon = model_output
        elif self.objective == "pred_v":
            x_recon = self.predict_start_from_v(x, t, model_output)
        else:
            raise ValueError(f"unknown objective {self.objective}")

        if clip_denoised:
            x_recon = x_recon.clamp(-1.0, 1.0)

        model_mean, model_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, model_log_variance, x_recon

    @torch.no_grad()
    def p_sample(self, x: torch.Tensor, t: torch.Tensor, clip_denoised: bool = True):
        model_mean, model_log_variance, x_recon = self.p_mean_variance(x, t, clip_denoised)
        noise = torch.randn_like(x)
        nonzero_mask = (t != 0).float().view(x.shape[0], *((1,) * (x.dim() - 1)))
        sample = model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise
        return sample, x_recon

    @torch.no_grad()
    def sample(self, batch_size: int = 1, return_all_timesteps: bool = False):
        device = self.betas.device
        shape = (batch_size, self.channels, self.image_size, self.image_size)
        img = torch.randn(shape, device=device)
        history = [img]

        for i in reversed(range(self.num_timesteps)):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            img, _ = self.p_sample(img, t)
            history.append(img)

        image = torch.stack(history, dim=1) if return_all_timesteps else img
        return self.unnormalize(image)

    def loss_func(self, pred: torch.Tensor, target: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss.mean(dim=(1,2,3))
        loss = loss * extract(self.loss_weight, t, loss.shape)
        return loss.mean()

    def p_losses(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        model_out = self.model(x_noisy, t)
        if isinstance(model_out, UNetOutput):
            model_out = model_out.pred

        if self.objective == "pred_noise":
            target = noise
        elif self.objective == "pred_x0":
            target = x_start
        elif self.objective == "pred_v":
            target = self.predict_v(x_start, t, noise)
        else:
            raise ValueError(f"unknown objective {self.objective}")

        return self.loss_func(model_out, target, t)

    def forward(self, x: torch.Tensor):
        b = x.shape[0]
        t = torch.randint(0, self.num_timesteps, (b,), device=x.device).long()
        x = self.normalize(x)
        return self.p_losses(x, t)

    @torch.no_grad()
    def feats(
        self,
        x_start: torch.Tensor,
        t: Union[int, torch.Tensor],
        noise: Optional[torch.Tensor] = None,
        feat_indices: Optional[Dict[str, Sequence[int]]] = None,
    ) -> Dict[str, Union[torch.Tensor, List[torch.Tensor]]]:
        if isinstance(t, int):
            t = torch.full((x_start.shape[0],), t, device=x_start.device, dtype=torch.long)

        x_start = self.normalize(x_start)
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        out = self.model(x_noisy, t, feat_need=True, feat_indices=feat_indices)
        return {
            "pred": out.pred,
            "encoder_feats": out.encoder_feats,
            "decoder_feats": out.decoder_feats,
            "mid_feat": out.mid_feat,
            "x_noisy": x_noisy,
            "t": t,
        }

    @torch.no_grad()
    def multi_t_feats(
        self,
        x_start: torch.Tensor,
        t_list: Sequence[int],
        noise: Optional[torch.Tensor] = None,
        feat_indices: Optional[Dict[str, Sequence[int]]] = None,
        stack: bool = False,
    ):
        results = {}
        x_start = self.normalize(x_start)
        base_noise = default(noise, lambda: torch.randn_like(x_start))

        for t in t_list:
            results[t] = self.feats(
                x_start=x_start,
                t=t,
                noise=base_noise,
                feat_indices=feat_indices,
            )

        if not stack:
            return results

        stacked = {}
        keys = results[t_list[0]].keys()
        for key in keys:
            vals = [results[t][key] for t in t_list]
            if isinstance(vals[0], list):
                per_level = []
                for lvl in range(len(vals[0])):
                    per_level.append(torch.stack([vals_t[lvl] for vals_t in vals], dim=1))
                stacked[key] = per_level
            elif torch.is_tensor(vals[0]):
                stacked[key] = torch.stack(vals, dim=1)
            else:
                stacked[key] = vals
        return stacked