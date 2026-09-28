import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange
import math
import warnings
from torch.nn.init import _calculate_fan_in_and_fan_out
import numbers

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def variance_scaling_(tensor, scale=1.0, mode='fan_in', distribution='normal'):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    if mode == 'fan_in':
        denom = fan_in
    elif mode == 'fan_out':
        denom = fan_out
    elif mode == 'fan_avg':
        denom = (fan_in + fan_out) / 2
    variance = scale / denom
    if distribution == "truncated_normal":
        trunc_normal_(tensor, std=math.sqrt(variance) / .87962566103423978)
    elif distribution == "normal":
        tensor.normal_(std=math.sqrt(variance))
    elif distribution == "uniform":
        bound = math.sqrt(3 * variance)
        tensor.uniform_(-bound, bound)
    else:
        raise ValueError(f"invalid distribution {distribution}")


def lecun_normal_(tensor):
    variance_scaling_(tensor, mode='fan_in', distribution='truncated_normal')



class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)


class Spectral_Atten(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.num_heads = heads
        self.to_q = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.to_k = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.to_v = nn.Conv2d(dim, dim, kernel_size=1, bias=False)

        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=False)
        self.k_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=False)
        self.v_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
    def forward(self, x_in):
        """
        x_in: [b,h,w,c]
        return out: [b,h,w,c]
        """
        b, c, h, w = x_in.shape
        q_in = self.q_dwconv(self.to_q(x_in))
        k_in = self.k_dwconv(self.to_k(x_in))
        v_in = self.v_dwconv(self.to_v(x_in))

        q = rearrange(q_in, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k_in, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v_in, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
  
        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)
        atten = (q @ k.transpose(-2, -1)) * self.rescale
        atten = atten.softmax(dim=-1)
        out = (atten @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.proj(out)

        return out


class Window_Spatial_Atten(nn.Module):
    """
    Local spatial self-attention inside non-overlapping windows.

    The window size is fixed (default 8x8), but the number of windows is
    computed dynamically from the current feature-map H/W. Therefore the
    same model can be trained with 512x512 images and directly tested with
    2048x2048 images. If H or W is not divisible by window_size, the feature
    map is padded on the right/bottom and cropped back after attention.
    """
    def __init__(self, dim, heads, window_size=8):
        super().__init__()
        assert dim % heads == 0, \
            f"dim ({dim}) must be divisible by heads ({heads})"

        self.num_heads = heads
        self.window_size = window_size
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5

        self.to_q = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.to_k = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.to_v = nn.Conv2d(dim, dim, kernel_size=1, bias=False)

        # Keep the same local inductive bias as the original spectral branch.
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=False)
        self.k_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=False)
        self.v_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=False)

        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=False)

    def _window_partition(self, x):
        # x: [B, C, Hp, Wp] -> [B*nH*nW, heads, ws*ws, head_dim]
        ws = self.window_size
        return rearrange(
            x,
            'b (head d) (nh wh) (nw ww) -> (b nh nw) head (wh ww) d',
            head=self.num_heads, wh=ws, ww=ws
        )

    def forward(self, x_in):
        """
        x_in: [B, C, H, W]
        return: [B, C, H, W]
        """
        b, c, h, w = x_in.shape
        ws = self.window_size

        # Dynamic padding makes the module resolution-agnostic.
        pad_h = (ws - h % ws) % ws
        pad_w = (ws - w % ws) % ws
        if pad_h != 0 or pad_w != 0:
            x = F.pad(x_in, (0, pad_w, 0, pad_h), mode='constant', value=0)
        else:
            x = x_in

        hp, wp = x.shape[-2:]
        nh, nw = hp // ws, wp // ws

        q = self.q_dwconv(self.to_q(x))
        k = self.k_dwconv(self.to_k(x))
        v = self.v_dwconv(self.to_v(x))

        q = self._window_partition(q)
        k = self._window_partition(k)
        v = self._window_partition(v)

        # Spatial attention is ws^2 x ws^2 inside each local window.
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = attn @ v

        out = rearrange(
            out,
            '(b nh nw) head (wh ww) d -> b (head d) (nh wh) (nw ww)',
            b=b, nh=nh, nw=nw, wh=ws, ww=ws
        )
        out = self.proj(out)

        # Remove dynamic padding.
        return out[:, :, :h, :w]


class Parallel_Spectral_Spatial_Atten(nn.Module):
    """Parallel CxC spectral attention + local 8x8 spatial attention."""
    def __init__(self, dim, heads, window_size=8):
        super().__init__()
        self.spectral_attn = Spectral_Atten(dim=dim, heads=heads)
        self.spatial_attn = Window_Spatial_Atten(
            dim=dim, heads=heads, window_size=window_size
        )

        # Start spatial branch conservatively so the optimization remains close
        # to the original spectral-only SAM at the beginning of training.
        self.spatial_weight = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        spectral = self.spectral_attn(x)
        spatial = self.spatial_attn(x)
        return spectral + self.spatial_weight * spatial


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super(LayerNorm, self).__init__()
        self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class GDFN(nn.Module):
    """
    Gated-Dconv Feed-Forward Network (GDFN), following Restormer.

    Structure:
        1x1 point-wise conv for channel expansion
        -> 3x3 depth-wise conv
        -> split into two branches
        -> GELU(x1) * x2 gated interaction
        -> 1x1 point-wise conv for channel projection
    """
    def __init__(self, dim, ffn_expansion_factor=2.66, bias=False):
        super().__init__()
        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(
            dim, hidden_features * 2, kernel_size=1, bias=bias
        )
        self.dwconv = nn.Conv2d(
            hidden_features * 2, hidden_features * 2, kernel_size=3,
            stride=1, padding=1, groups=hidden_features * 2, bias=bias
        )
        self.project_out = nn.Conv2d(
            hidden_features, dim, kernel_size=1, bias=bias
        )

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class SAM_Spectral(nn.Module):
    def __init__(self, dim, heads, num_blocks, window_size=8, ffn_expansion_factor=2.66):
        super().__init__()

        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                LayerNorm(dim),
                Parallel_Spectral_Spatial_Atten(
                    dim=dim, heads=heads, window_size=window_size
                ),
                LayerNorm(dim),
                GDFN(dim=dim, ffn_expansion_factor=ffn_expansion_factor, bias=False)
            ]))

    def forward(self, x):
        """
        x: [b,c,h,w]
        return out: [b,c,h,w]

        Each block uses parallel attention followed by Restormer GDFN:
            CxC spectral attention || 8x8 local spatial attention
            -> GDFN
        """
        for (norm1, atten, norm2, ffn) in self.blocks:
            x = atten(norm1(x)) + x
            x = ffn(norm2(x)) + x
        return x

class SRNet_parallel_spectral_GDFN(nn.Module):
    def __init__(self, in_channels=1, out_channels=61, dim=32, deep_stage=3, num_blocks=[1, 1, 1], num_heads=[1, 2, 4], window_size=8, ffn_expansion_factor=2.66):
        super(SRNet_parallel_spectral_GDFN, self).__init__()
        self.dim = dim
        self.out_channels = out_channels
        self.stage = deep_stage
        self.window_size = window_size

        self.embedding1 = nn.Conv2d(in_channels, dim, kernel_size=3, padding=1, bias=False)
        self.embedding2 = nn.Conv2d(out_channels, dim, kernel_size=3, padding=1, bias=False)
        self.embedding = nn.Conv2d(dim * 2, dim, kernel_size=3, padding=1, bias=False)
        
        self.down_sample = nn.Conv2d(dim, dim, 4, 2, 1, bias=False)
        self.up_sample = nn.ConvTranspose2d(dim, dim, stride=2, kernel_size=2, padding=0, output_padding=0)

        self.mapping = nn.Conv2d(dim, out_channels, kernel_size=3, padding=1, bias=False)


        self.encoder_layers = nn.ModuleList([])
        dim_stage = dim
        for i in range(deep_stage):
            self.encoder_layers.append(nn.ModuleList([
                SAM_Spectral(dim=dim_stage, heads=num_heads[i], num_blocks=num_blocks[i], window_size=window_size, ffn_expansion_factor=ffn_expansion_factor),
                nn.Conv2d(dim_stage, dim_stage * 2, 4, 2, 1, bias=False),
            ]))
            dim_stage *= 2


        self.bottleneck = SAM_Spectral(
            dim=dim_stage, heads=num_heads[-1], num_blocks=num_blocks[-1], window_size=window_size, ffn_expansion_factor=ffn_expansion_factor)

        self.decoder_layers = nn.ModuleList([])
        for i in range(deep_stage):
            self.decoder_layers.append(nn.ModuleList([
                nn.ConvTranspose2d(dim_stage, dim_stage // 2, stride=2, kernel_size=2, padding=0, output_padding=0),
                nn.Conv2d(dim_stage, dim_stage // 2, 1, 1, bias=False),
                SAM_Spectral(dim=dim_stage // 2, heads=num_heads[deep_stage - 1 - i], num_blocks=num_blocks[deep_stage - 1 - i], window_size=window_size, ffn_expansion_factor=ffn_expansion_factor),
            ]))
            dim_stage //= 2

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, mask, return_feat=False):
        """
        x: [b,c,h,w]
        mask: [b,out_channels,h,w]
        return_feat=False: return out
        return_feat=True : return out, bottleneck feature
        """

        x = self.embedding1(x)
        mask = self.embedding2(mask)
        x = torch.cat((x, mask), dim=1)

        fea = self.embedding(x)
        residual = fea
        fea = self.down_sample(fea)

        fea_encoder = []
        for (Attention, FeaDownSample) in self.encoder_layers:
            fea = Attention(fea)
            fea_encoder.append(fea)
            fea = FeaDownSample(fea)

    
        fea = self.bottleneck(fea)
        bottleneck_feat = fea
 
        for i, (FeaUpSample, Fution, Attention) in enumerate(self.decoder_layers):
            fea = FeaUpSample(fea)
            fea = Fution(torch.cat([fea, fea_encoder[self.stage - 1 - i]], dim=1))
            fea = Attention(fea)
 
        fea = self.up_sample(fea)
        out = fea + residual
        out = self.mapping(out)

        if return_feat:
            return out, bottleneck_feat
        else:
            return out














# Backward-compatible alias for existing training scripts
SRNet_parallel_spectral = SRNet_parallel_spectral_GDFN
