import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def exists(x):
    return x is not None


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 2:
            t = t.squeeze(-1)
        t = t.float()
        half_dim = self.dim // 2
        emb_scale = math.log(10000) / max(half_dim - 1, 1)
        emb = torch.exp(
            torch.arange(half_dim, device=t.device, dtype=t.dtype) * -emb_scale
        )
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class FeatureWiseAffine(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, use_affine_level: bool = True):
        super().__init__()
        self.use_affine_level = use_affine_level
        self.time_func = nn.Linear(in_dim, out_dim * (2 if use_affine_level else 1))

    def forward(self, x: torch.Tensor, time_emb: Optional[torch.Tensor]) -> torch.Tensor:
        if time_emb is None:
            return x
        style = self.time_func(time_emb).view(x.size(0), -1, 1, 1)
        if self.use_affine_level:
            gamma, beta = style.chunk(2, dim=1)
            return (1 + gamma) * x + beta
        return x + style


class Block(nn.Module):
    def __init__(self, dim: int, dim_out: int, groups: int = 32, dropout: float = 0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(groups, dim),
            Swish(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(dim, dim_out, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResnetBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int,
        time_emb_dim: Optional[int] = None,
        groups: int = 32,
        dropout: float = 0.0,
        use_affine_level: bool = True,
    ):
        super().__init__()
        self.time_func = (
            FeatureWiseAffine(time_emb_dim, dim_out, use_affine_level)
            if exists(time_emb_dim)
            else None
        )
        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups, dropout=dropout)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x: torch.Tensor, time_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.block1(x)
        if exists(self.time_func):
            h = self.time_func(h, time_emb)
        h = self.block2(h)
        return h + self.res_conv(x)


class LinearAttention2d(nn.Module):
    def __init__(self, dim: int, heads: int = 4, dim_head: int = 32, norm_groups: int = 32):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        hidden_dim = heads * dim_head
        self.norm = nn.GroupNorm(norm_groups, dim)
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        x_norm = self.norm(x)
        qkv = self.to_qkv(x_norm)
        q, k, v = qkv.chunk(3, dim=1)

        q = q.view(b, self.heads, self.dim_head, h * w)
        k = k.view(b, self.heads, self.dim_head, h * w)
        v = v.view(b, self.heads, self.dim_head, h * w)

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)
        q = q * (self.dim_head ** -0.5)

        context = torch.einsum("bhdn,bhen->bhde", k, v)
        out = torch.einsum("bhde,bhdn->bhen", context, q)
        out = out.contiguous().view(b, self.heads * self.dim_head, h, w)
        out = self.to_out(out)
        return x + out


class FullAttention2d(nn.Module):
    def __init__(self, dim: int, heads: int = 4, dim_head: int = 32, norm_groups: int = 32):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        hidden_dim = heads * dim_head
        self.norm = nn.GroupNorm(norm_groups, dim)
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        x_norm = self.norm(x)
        qkv = self.to_qkv(x_norm)
        q, k, v = qkv.chunk(3, dim=1)

        q = q.view(b, self.heads, self.dim_head, h * w).permute(0, 1, 3, 2)
        k = k.view(b, self.heads, self.dim_head, h * w)
        v = v.view(b, self.heads, self.dim_head, h * w).permute(0, 1, 3, 2)

        attn = torch.matmul(q, k) * (self.dim_head ** -0.5)
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        out = out.permute(0, 1, 3, 2).contiguous().view(b, self.heads * self.dim_head, h, w)
        out = self.to_out(out)
        return x + out


class ResnetBlockWithAttn(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int,
        time_emb_dim: Optional[int],
        norm_groups: int,
        dropout: float,
        attn_type: str = "none",
        attn_heads: int = 4,
        attn_dim_head: int = 32,
    ):
        super().__init__()
        self.res_block = ResnetBlock(
            dim=dim,
            dim_out=dim_out,
            time_emb_dim=time_emb_dim,
            groups=norm_groups,
            dropout=dropout,
        )

        if attn_type == "none":
            self.attn = nn.Identity()
        elif attn_type == "linear":
            self.attn = LinearAttention2d(
                dim_out, heads=attn_heads, dim_head=attn_dim_head, norm_groups=norm_groups
            )
        elif attn_type == "full":
            self.attn = FullAttention2d(
                dim_out, heads=attn_heads, dim_head=attn_dim_head, norm_groups=norm_groups
            )
        else:
            raise ValueError(f"unsupported attn_type: {attn_type}")

    def forward(self, x: torch.Tensor, time_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.res_block(x, time_emb)
        x = self.attn(x)
        return x


class Downsample(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.op = nn.Conv2d(dim, dim, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.op = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.op(x)


@dataclass
class UNetOutput:
    pred: Optional[torch.Tensor]
    encoder_feats: Optional[List[torch.Tensor]] = None
    decoder_feats: Optional[List[torch.Tensor]] = None
    mid_feat: Optional[torch.Tensor] = None


class UNet(nn.Module):
    def __init__(
        self,
        in_channel: int = 61,
        out_channel: int = 61,
        inner_channel: int = 64,
        norm_groups: int = 32,
        channel_mults: Sequence[int] = (1, 2, 4, 8),
        res_blocks: int = 2,
        dropout: float = 0.0,
        with_time_emb: bool = True,
        stage_attn_types: Optional[Sequence[str]] = None,
        mid_attn_type: str = "full",
        attn_heads: int = 4,
        attn_dim_head: int = 32,
    ):
        super().__init__()
        self.out_channel = out_channel

        if with_time_emb:
            time_emb_dim = inner_channel
            self.time_mlp = nn.Sequential(
                SinusoidalTimeEmbedding(inner_channel),
                nn.Linear(inner_channel, inner_channel * 4),
                Swish(),
                nn.Linear(inner_channel * 4, inner_channel),
            )
        else:
            time_emb_dim = None
            self.time_mlp = None

        self.init_conv = nn.Conv2d(in_channel, inner_channel, 3, padding=1)

        if stage_attn_types is None:
            stage_attn_types = tuple(["linear"] * (len(channel_mults) - 1) + ["full"])
        assert len(stage_attn_types) == len(channel_mults)

        downs = []
        feat_channels = [inner_channel]
        pre_channel = inner_channel

        for i, mult in enumerate(channel_mults):
            out_ch = inner_channel * mult
            attn_type = stage_attn_types[i]
            for _ in range(res_blocks):
                downs.append(
                    ResnetBlockWithAttn(
                        pre_channel,
                        out_ch,
                        time_emb_dim,
                        norm_groups,
                        dropout,
                        attn_type=attn_type,
                        attn_heads=attn_heads,
                        attn_dim_head=attn_dim_head,
                    )
                )
                pre_channel = out_ch
                feat_channels.append(pre_channel)

            if i != len(channel_mults) - 1:
                downs.append(Downsample(pre_channel))
                feat_channels.append(pre_channel)

        self.downs = nn.ModuleList(downs)

        self.mid = nn.ModuleList([
            ResnetBlockWithAttn(
                pre_channel, pre_channel, time_emb_dim, norm_groups, dropout,
                attn_type=mid_attn_type,
                attn_heads=attn_heads,
                attn_dim_head=attn_dim_head,
            ),
            ResnetBlockWithAttn(
                pre_channel, pre_channel, time_emb_dim, norm_groups, dropout,
                attn_type="none",
                attn_heads=attn_heads,
                attn_dim_head=attn_dim_head,
            ),
        ])

        ups = []
        for i, mult in reversed(list(enumerate(channel_mults))):
            out_ch = inner_channel * mult
            attn_type = stage_attn_types[i]
            for _ in range(res_blocks + 1):
                skip_ch = feat_channels.pop()
                ups.append(
                    ResnetBlockWithAttn(
                        pre_channel + skip_ch,
                        out_ch,
                        time_emb_dim,
                        norm_groups,
                        dropout,
                        attn_type=attn_type,
                        attn_heads=attn_heads,
                        attn_dim_head=attn_dim_head,
                    )
                )
                pre_channel = out_ch

            if i != 0:
                ups.append(Upsample(pre_channel))

        self.ups = nn.ModuleList(ups)

        self.final_conv = nn.Sequential(
            Block(pre_channel, pre_channel, groups=norm_groups),
            nn.Conv2d(pre_channel, out_channel, 3, padding=1),
        )

    def forward(
        self,
        x: torch.Tensor,
        time: torch.Tensor,
        feat_need: bool = False,
        feat_indices: Optional[Dict[str, Sequence[int]]] = None,
    ) -> Union[torch.Tensor, UNetOutput]:
        if time.dim() == 0:
            time = time[None]
        if time.dim() == 2 and time.shape[1] == 1:
            time = time.squeeze(-1)

        t = self.time_mlp(time) if exists(self.time_mlp) else None

        x = self.init_conv(x)
        feats = [x]

        encoder_feats = [] if feat_need else None

        for layer in self.downs:
            if isinstance(layer, ResnetBlockWithAttn):
                x = layer(x, t)
                feats.append(x)
                if feat_need:
                    encoder_feats.append(x)
            else:
                x = layer(x)
                feats.append(x)

        for layer in self.mid:
            x = layer(x, t)
        mid_feat = x

        decoder_feats = [] if feat_need else None
        for layer in self.ups:
            if isinstance(layer, ResnetBlockWithAttn):
                skip = feats.pop()
                x = torch.cat([x, skip], dim=1)
                x = layer(x, t)
                if feat_need:
                    decoder_feats.append(x)
            else:
                x = layer(x)

        pred = self.final_conv(x)

        if feat_need:
            if decoder_feats is not None:
                decoder_feats = list(reversed(decoder_feats))

            if feat_indices is not None:
                if encoder_feats is not None and "encoder" in feat_indices:
                    encoder_feats = [encoder_feats[i] for i in feat_indices["encoder"]]
                if decoder_feats is not None and "decoder" in feat_indices:
                    decoder_feats = [decoder_feats[i] for i in feat_indices["decoder"]]

            return UNetOutput(
                pred=pred,
                encoder_feats=encoder_feats,
                decoder_feats=decoder_feats,
                mid_feat=mid_feat,
            )

        return pred