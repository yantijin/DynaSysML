import math
import copy
import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from inspect import isfunction
from einops import rearrange
import numpy as np
try:
    from apex import amp
    APEX_AVAILABLE = True
except:
    APEX_AVAILABLE = False

# helpers functions

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def cycle(dl):
    while True:
        for data in dl:
            yield data

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def loss_backwards(fp16, loss, optimizer, **kwargs):
    if fp16:
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward(**kwargs)
    else:
        loss.backward(**kwargs)

# small helper modules

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class GaussianFourierProjection(nn.Module):
  """Gaussian Fourier embeddings for noise levels."""

  def __init__(self, embedding_size=256, scale=1.0):
    super().__init__()
    self.W = nn.Parameter(torch.randn(embedding_size) * scale, requires_grad=False)

  def forward(self, x):
    x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
    return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose2d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)

class LayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        std = torch.var(x, dim = 1, unbiased = False, keepdim = True).sqrt()
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (std + self.eps) * self.g + self.b

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

# building block modules

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups = 8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim_out, 3, padding=1),
            nn.GroupNorm(groups, dim_out),
            Mish()
        )
    def forward(self, x):
        return self.block(x)

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, groups = 8):
        super().__init__()
        self.mlp = nn.Sequential(
            Mish(),
            nn.Linear(time_emb_dim, dim_out)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out)
        self.block2 = Block(dim_out, dim_out)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb):
        h = self.block1(x)

        if exists(self.mlp):
            h += self.mlp(time_emb)[:, :, None, None]

        h = self.block2(h)
        return h + self.res_conv(x)

class LinearAttention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads = self.heads, qkv=3)
        k = k.softmax(dim=-1)
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)
        return self.to_out(out)

# model

class Unet(nn.Module):
    def __init__(
        self,
        dim,
        out_dim = None,
        dim_mults=(1, 2, 4, 8),
        groups = 8,
        channels = 3,
        with_time_emb = True
    ):
        super().__init__()
        self.channels = channels

        dims = [channels, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        if with_time_emb:
            time_dim = dim
            self.time_mlp = nn.Sequential(
                SinusoidalPosEmb(dim),
                nn.Linear(dim, dim * 4),
                Mish(),
                nn.Linear(dim * 4, dim)
            )
        else:
            time_dim = None
            self.time_mlp = None

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                ResnetBlock(dim_in, dim_out, time_emb_dim = time_dim),
                ResnetBlock(dim_out, dim_out, time_emb_dim = time_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Downsample(dim_out) if not is_last else nn.Identity()
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim = time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, LinearAttention(mid_dim)))
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim = time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                ResnetBlock(dim_out * 2, dim_in, time_emb_dim = time_dim),
                ResnetBlock(dim_in, dim_in, time_emb_dim = time_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Upsample(dim_in) if not is_last else nn.Identity()
            ]))

        out_dim = default(out_dim, channels)
        self.final_conv = nn.Sequential(
            Block(dim, dim),
            nn.Conv2d(dim, out_dim, 1)
        )

    def forward(self, x, t):
        t = self.time_mlp(t) if exists(self.time_mlp) else None

        h = []

        for resnet, resnet2, attn, downsample in self.downs:
            x = resnet(x, t)
            x = resnet2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for resnet, resnet2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, t)
            x = resnet2(x, t)
            x = attn(x)
            x = upsample(x)

        return self.final_conv(x)


class Cond_ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, cond_dim=None, groups = 8):
        super().__init__()
        self.mlp = nn.Sequential(
            Mish(),
            nn.Linear(time_emb_dim, dim_out)
        ) if exists(time_emb_dim) else None
        self.mlp1 = nn.Sequential(
            Mish(),
            nn.Linear(cond_dim, dim_out)
        ) if exists(cond_dim) else None

        self.block1 = Block(dim, dim_out)
        self.block2 = Block(dim_out, dim_out)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, cond_emb, time_emb):
        h = self.block1(x)

        if exists(self.mlp):
            h += self.mlp(time_emb)[:, :, None, None]

        if exists(self.mlp1):
            h += self.mlp1(cond_emb)[:, :, None, None]

        h = self.block2(h)
        return h + self.res_conv(x)


class Cond_Unet(nn.Module):
    def __init__(
        self,
        dim,
        out_dim = None,
        dim_mults=(1, 2, 4, 8),
        groups = 8,
        channels = 3,
        with_time_emb = True,
        with_cond_emb = True
    ):
        super().__init__()
        self.channels = channels

        dims = [channels, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        if with_time_emb:
            time_dim = dim
            self.time_mlp = nn.Sequential(
                SinusoidalPosEmb(dim),
                nn.Linear(dim, dim * 4),
                Mish(),
                nn.Linear(dim * 4, dim)
            )
        else:
            time_dim = None
            self.time_mlp = None

        if with_cond_emb:
            self.cond_mlp = nn.Sequential(
                SinusoidalPosEmb(dim),
                nn.Linear(dim, dim * 4),
                Mish(),
                nn.Linear(dim * 4, dim)
            )
        else:
            self.cond_mlp = None

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                Cond_ResnetBlock(dim_in, dim_out, time_emb_dim = time_dim, cond_dim=dim),
                Cond_ResnetBlock(dim_out, dim_out, time_emb_dim = time_dim, cond_dim=dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Downsample(dim_out) if not is_last else nn.Identity()
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = Cond_ResnetBlock(mid_dim, mid_dim, time_emb_dim = time_dim, cond_dim=dim)
        self.mid_attn = Residual(PreNorm(mid_dim, LinearAttention(mid_dim)))
        self.mid_block2 = Cond_ResnetBlock(mid_dim, mid_dim, time_emb_dim = time_dim, cond_dim=dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                Cond_ResnetBlock(dim_out * 2, dim_in, time_emb_dim = time_dim, cond_dim=dim),
                Cond_ResnetBlock(dim_in, dim_in, time_emb_dim = time_dim, cond_dim=dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Upsample(dim_in) if not is_last else nn.Identity()
            ]))

        out_dim = default(out_dim, channels)
        self.final_conv = nn.Sequential(
            Block(dim, dim),
            nn.Conv2d(dim, out_dim, 1)
        )

    def forward(self, x, t, cond):
        t = self.time_mlp(t) if exists(self.time_mlp) else None
        cond = self.cond_mlp(cond) if exists(self.cond_mlp) else None

        h = []

        for resnet, resnet2, attn, downsample in self.downs:
            x = resnet(x, cond, t)
            x = resnet2(x, cond, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, cond, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, cond, t)

        for resnet, resnet2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, cond, t)
            x = resnet2(x, cond, t)
            x = attn(x)
            x = upsample(x)

        return self.final_conv(x)


class cont_Unet(nn.Module):
    def __init__(
        self,
        dim,
        out_dim = None,
        dim_mults=(1, 2, 4, 8),
        groups = 8,
        channels = 3,
        with_time_emb = True
    ):
        super().__init__()
        self.channels = channels

        dims = [channels, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        if with_time_emb:
            time_dim = dim
            self.time_mlp = nn.Sequential(
                GaussianFourierProjection(dim//2),
                nn.Linear(dim, dim * 4),
                Mish(),
                nn.Linear(dim * 4, dim)
            )
        else:
            time_dim = None
            self.time_mlp = None

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                ResnetBlock(dim_in, dim_out, time_emb_dim = time_dim),
                ResnetBlock(dim_out, dim_out, time_emb_dim = time_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Downsample(dim_out) if not is_last else nn.Identity()
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim = time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, LinearAttention(mid_dim)))
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim = time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                ResnetBlock(dim_out * 2, dim_in, time_emb_dim = time_dim),
                ResnetBlock(dim_in, dim_in, time_emb_dim = time_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Upsample(dim_in) if not is_last else nn.Identity()
            ]))

        out_dim = default(out_dim, channels)
        self.final_conv = nn.Sequential(
            Block(dim, dim),
            nn.Conv2d(dim, out_dim, 1)
        )

    def forward(self, x, t):
        t = self.time_mlp(t) if exists(self.time_mlp) else None

        h = []

        for resnet, resnet2, attn, downsample in self.downs:
            x = resnet(x, t)
            x = resnet2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for resnet, resnet2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, t)
            x = resnet2(x, t)
            x = attn(x)
            x = upsample(x)

        return self.final_conv(x)


class logistic_Unet(nn.Module):
    def __init__(
        self,
        dim,
        out_dim = None,
        dim_mults=(1, 2, 4, 8),
        groups = 8,
        channels = 3,
        with_time_emb = True
    ):
        super().__init__()
        self.channels = channels

        dims = [channels, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        if with_time_emb:
            time_dim = dim
            self.time_mlp = nn.Sequential(
                SinusoidalPosEmb(dim),
                nn.Linear(dim, dim * 4),
                Mish(),
                nn.Linear(dim * 4, dim)
            )
        else:
            time_dim = None
            self.time_mlp = None

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                ResnetBlock(dim_in, dim_out, time_emb_dim = time_dim),
                ResnetBlock(dim_out, dim_out, time_emb_dim = time_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Downsample(dim_out) if not is_last else nn.Identity()
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim = time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, LinearAttention(mid_dim)))
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim = time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                ResnetBlock(dim_out * 2, dim_in, time_emb_dim = time_dim),
                ResnetBlock(dim_in, dim_in, time_emb_dim = time_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Upsample(dim_in) if not is_last else nn.Identity()
            ]))

        out_dim = default(out_dim, channels)
        self.final_conv = nn.Sequential(
            Block(dim, dim),
            nn.Conv2d(dim, out_dim*2, 1)
        )

    def forward(self, x, t):
        x = x/255. # normalize data
        t = self.time_mlp(t) if exists(self.time_mlp) else None

        h = []

        for resnet, resnet2, attn, downsample in self.downs:
            x = resnet(x, t)
            x = resnet2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for resnet, resnet2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, t)
            x = resnet2(x, t)
            x = attn(x)
            x = upsample(x)

        res = self.final_conv(x)
        loc = torch.sigmoid(res[:, :self.channels])
        log_scale = res[:, self.channels:]
        return loc, log_scale


class Cond_Gen(nn.Module):
    def __init__(
        self,
        dim,
        out_dim = None,
        dim_mults=(1, 2, 4, 8),
        groups = 8,
        latent_dim=100,
        channels = 3,
        with_time_emb = True,
        with_cond_emb = True
    ):
        super().__init__()
        self.channels = channels

        dims = [channels, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        if with_time_emb:
            time_dim = dim
            self.time_mlp = nn.Sequential(
                SinusoidalPosEmb(dim),
                nn.Linear(dim, dim * 4),
                Mish(),
                nn.Linear(dim * 4, dim)
            )
        else:
            time_dim = None
            self.time_mlp = None

        if with_cond_emb:
            self.cond_mlp = nn.Sequential(
                nn.Linear(latent_dim, dim * 4),
                Mish(),
                nn.Linear(dim * 4, dim)
            )
        else:
            self.cond_mlp = None

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                Cond_ResnetBlock(dim_in, dim_out, time_emb_dim = time_dim, cond_dim=dim),
                Cond_ResnetBlock(dim_out, dim_out, time_emb_dim = time_dim, cond_dim=dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Downsample(dim_out) if not is_last else nn.Identity()
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = Cond_ResnetBlock(mid_dim, mid_dim, time_emb_dim = time_dim, cond_dim=dim)
        self.mid_attn = Residual(PreNorm(mid_dim, LinearAttention(mid_dim)))
        self.mid_block2 = Cond_ResnetBlock(mid_dim, mid_dim, time_emb_dim = time_dim, cond_dim=dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                Cond_ResnetBlock(dim_out * 2, dim_in, time_emb_dim = time_dim, cond_dim=dim),
                Cond_ResnetBlock(dim_in, dim_in, time_emb_dim = time_dim, cond_dim=dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Upsample(dim_in) if not is_last else nn.Identity()
            ]))

        out_dim = default(out_dim, channels)
        self.final_conv = nn.Sequential(
            Block(dim, dim),
            nn.Conv2d(dim, out_dim, 1)
        )

    def forward(self, x, t, cond):
        t = self.time_mlp(t) if exists(self.time_mlp) else None
        cond = self.cond_mlp(cond) if exists(self.cond_mlp) else None

        h = []

        for resnet, resnet2, attn, downsample in self.downs:
            x = resnet(x, cond, t)
            x = resnet2(x, cond, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, cond, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, cond, t)

        for resnet, resnet2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, cond, t)
            x = resnet2(x, cond, t)
            x = attn(x)
            x = upsample(x)

        return self.final_conv(x)

class Dis(nn.Module):
    def __init__(
            self,
            dim,
            out_dim=None,
            dim_mults=(1, 2, 4, 8),
            groups=8,
            channels=3,
            with_time_emb=True
    ):
        super().__init__()
        self.channels = channels

        dims = [channels, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        if with_time_emb:
            time_dim = dim
            self.time_mlp = nn.Sequential(
                SinusoidalPosEmb(dim),
                nn.Linear(dim, dim * 4),
                Mish(),
                nn.Linear(dim * 4, dim)
            )
        else:
            time_dim = None
            self.time_mlp = None

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                ResnetBlock(dim_in, dim_out, time_emb_dim=time_dim),
                ResnetBlock(dim_out, dim_out, time_emb_dim=time_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Downsample(dim_out) if not is_last else nn.Identity()
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, LinearAttention(mid_dim)))
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.pool = nn.AvgPool2d(kernel_size=4)
        self.final_l = nn.Sequential(
            nn.Linear(dim*dim_mults[-1], 1),
            nn.Sigmoid()
        )

        # for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
        #     is_last = ind >= (num_resolutions - 1)
        #
        #     self.ups.append(nn.ModuleList([
        #         ResnetBlock(dim_out * 2, dim_in, time_emb_dim=time_dim),
        #         ResnetBlock(dim_in, dim_in, time_emb_dim=time_dim),
        #         Residual(PreNorm(dim_in, LinearAttention(dim_in))),
        #         Upsample(dim_in) if not is_last else nn.Identity()
        #     ]))
        #
        # out_dim = default(out_dim, channels)
        # self.final_conv = nn.Sequential(
        #     Block(dim, dim),
        #     nn.Conv2d(dim, out_dim, 1)
        # )

    def forward(self, xt, xt_1, t):
        batch = xt.shape[0]
        x = torch.cat([xt, xt_1], dim=1)
        t = self.time_mlp(t) if exists(self.time_mlp) else None

        h = []

        for resnet, resnet2, attn, downsample in self.downs:
            x = resnet(x, t)
            x = resnet2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)
        x = self.pool(x) * 4
        x = self.final_l(x.view(batch, -1))
        return x


class Cond_logistic_gen(nn.Module):
    def __init__(
        self,
        dim,
        out_dim = None,
        dim_mults=(1, 2, 4, 8),
        groups = 8,
        latent_dim=100,
        channels = 3,
        with_time_emb = True,
        with_cond_emb = True
    ):
        super().__init__()
        self.channels = channels

        dims = [channels, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        if with_time_emb:
            time_dim = dim
            self.time_mlp = nn.Sequential(
                SinusoidalPosEmb(dim),
                nn.Linear(dim, dim * 4),
                Mish(),
                nn.Linear(dim * 4, dim)
            )
        else:
            time_dim = None
            self.time_mlp = None

        if with_cond_emb:
            self.cond_mlp = nn.Sequential(
                nn.Linear(latent_dim, dim * 4),
                Mish(),
                nn.Linear(dim * 4, dim)
            )
        else:
            self.cond_mlp = None

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                Cond_ResnetBlock(dim_in, dim_out, time_emb_dim=time_dim, cond_dim=dim),
                Cond_ResnetBlock(dim_out, dim_out, time_emb_dim=time_dim, cond_dim=dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Downsample(dim_out) if not is_last else nn.Identity()
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = Cond_ResnetBlock(mid_dim, mid_dim, time_emb_dim=time_dim, cond_dim=dim)
        self.mid_attn = Residual(PreNorm(mid_dim, LinearAttention(mid_dim)))
        self.mid_block2 = Cond_ResnetBlock(mid_dim, mid_dim, time_emb_dim=time_dim, cond_dim=dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                Cond_ResnetBlock(dim_out * 2, dim_in, time_emb_dim=time_dim, cond_dim=dim),
                Cond_ResnetBlock(dim_in, dim_in, time_emb_dim=time_dim, cond_dim=dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Upsample(dim_in) if not is_last else nn.Identity()
            ]))

        out_dim = default(out_dim, channels)
        self.final_conv = nn.Sequential(
            Block(dim, dim),
            nn.Conv2d(dim, out_dim*2, 1)
        )

    def forward(self, x, t, cond):
        x = x/255. # normalize data
        t = self.time_mlp(t) if exists(self.time_mlp) else None
        cond = self.cond_mlp(cond) if exists(self.cond_mlp) else None

        h = []

        for resnet, resnet2, attn, downsample in self.downs:
            x = resnet(x, cond, t)
            x = resnet2(x, cond, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, cond, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, cond, t)

        for resnet, resnet2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, cond, t)
            x = resnet2(x, cond, t)
            x = attn(x)
            x = upsample(x)

        res = self.final_conv(x)
        loc = torch.sigmoid(res[:, :self.channels])
        log_scale = res[:, self.channels:]
        return loc, log_scale


class c_Gen(nn.Module):
    def __init__(
        self,
        dim,
        out_dim = None,
        dim_mults=(1, 2, 4, 8),
        groups = 8,
        latent_dim=100,
        channels = 3,
        num_classes=10,
        with_time_emb = True,
        with_cond_emb = True,
    ):
        super().__init__()
        self.channels = channels + num_classes

        dims = [channels+num_classes, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        if with_time_emb:
            time_dim = dim
            self.time_mlp = nn.Sequential(
                SinusoidalPosEmb(dim),
                nn.Linear(dim, dim * 4),
                Mish(),
                nn.Linear(dim * 4, dim)
            )
        else:
            time_dim = None
            self.time_mlp = None

        if with_cond_emb:
            self.cond_mlp = nn.Sequential(
                nn.Linear(latent_dim+num_classes, dim * 4),
                Mish(),
                nn.Linear(dim * 4, dim)
            )
        else:
            self.cond_mlp = None

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                Cond_ResnetBlock(dim_in, dim_out, time_emb_dim = time_dim, cond_dim=dim),
                Cond_ResnetBlock(dim_out, dim_out, time_emb_dim = time_dim, cond_dim=dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Downsample(dim_out) if not is_last else nn.Identity()
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = Cond_ResnetBlock(mid_dim, mid_dim, time_emb_dim = time_dim, cond_dim=dim)
        self.mid_attn = Residual(PreNorm(mid_dim, LinearAttention(mid_dim)))
        self.mid_block2 = Cond_ResnetBlock(mid_dim, mid_dim, time_emb_dim = time_dim, cond_dim=dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                Cond_ResnetBlock(dim_out * 2, dim_in, time_emb_dim = time_dim, cond_dim=dim),
                Cond_ResnetBlock(dim_in, dim_in, time_emb_dim = time_dim, cond_dim=dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Upsample(dim_in) if not is_last else nn.Identity()
            ]))

        out_dim = default(out_dim, channels)
        self.final_conv = nn.Sequential(
            Block(dim, dim),
            nn.Conv2d(dim, out_dim, 1)
        )

    def forward(self, x, t, cond, label):
        label_emb = torch.nn.functional.one_hot(label, 10).float() # mnist一共有10类
        cond = torch.cat([cond, label_emb], dim=1) #[bs, cond_dim+10]
        label_emb = label_emb.unsqueeze(-1).unsqueeze(-1) #[bs, 10, 1, 1]
        x = torch.cat([x, label_emb*torch.ones_like(x)], dim=1) #[bs, c+10, 28, 28]
        t = self.time_mlp(t) if exists(self.time_mlp) else None
        cond = self.cond_mlp(cond) if exists(self.cond_mlp) else None
        h = []

        for resnet, resnet2, attn, downsample in self.downs:
            x = resnet(x, cond, t)
            x = resnet2(x, cond, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, cond, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, cond, t)

        for resnet, resnet2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, cond, t)
            x = resnet2(x, cond, t)
            x = attn(x)
            x = upsample(x)

        return self.final_conv(x)


class c_Dis(nn.Module):
    def __init__(
        self,
        dim,
        out_dim = None,
        dim_mults=(1, 2, 4, 8),
        groups = 8,
        latent_dim=100,
        channels = 3,
        num_classes=10,
        with_time_emb = True,
        with_cond_emb = True,
    ):
        super().__init__()
        self.channels = channels

        dims = [channels, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        if with_time_emb:
            time_dim = dim
            self.time_mlp = nn.Sequential(
                SinusoidalPosEmb(dim),
                nn.Linear(dim, dim * 4),
                Mish(),
                nn.Linear(dim * 4, dim)
            )
        else:
            time_dim = None
            self.time_mlp = None

        if with_cond_emb:
            self.cond_mlp = nn.Sequential(
                nn.Linear(num_classes, dim * 4),
                Mish(),
                nn.Linear(dim * 4, dim)
            )
        else:
            self.cond_mlp = None

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                Cond_ResnetBlock(dim_in, dim_out, time_emb_dim = time_dim, cond_dim=dim),
                Cond_ResnetBlock(dim_out, dim_out, time_emb_dim = time_dim, cond_dim=dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Downsample(dim_out) if not is_last else nn.Identity()
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = Cond_ResnetBlock(mid_dim, mid_dim, time_emb_dim = time_dim, cond_dim=dim)
        self.mid_attn = Residual(PreNorm(mid_dim, LinearAttention(mid_dim)))
        self.mid_block2 = Cond_ResnetBlock(mid_dim, mid_dim, time_emb_dim = time_dim, cond_dim=dim)
        self.pool = nn.AvgPool2d(kernel_size=4)
        self.final_l = nn.Sequential(
            nn.Linear(dim * dim_mults[-1], 1),
            nn.Sigmoid()
        )

        # for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
        #     is_last = ind >= (num_resolutions - 1)
        #
        #     self.ups.append(nn.ModuleList([
        #         Cond_ResnetBlock(dim_out * 2, dim_in, time_emb_dim = time_dim, cond_dim=dim),
        #         Cond_ResnetBlock(dim_in, dim_in, time_emb_dim = time_dim, cond_dim=dim),
        #         Residual(PreNorm(dim_in, LinearAttention(dim_in))),
        #         Upsample(dim_in) if not is_last else nn.Identity()
        #     ]))
        #
        # out_dim = default(out_dim, channels)
        # self.final_conv = nn.Sequential(
        #     Block(dim, dim),
        #     nn.Conv2d(dim, out_dim, 1)
        # )

    def forward(self, xt, xt_1, t, label):
        label_emb = torch.nn.functional.one_hot(label, 10).float() # [bs, 10]
        batch = xt.shape[0]
        x = torch.cat([xt, xt_1], dim=1)
        t = self.time_mlp(t) if exists(self.time_mlp) else None
        cond = self.cond_mlp(label_emb) if exists(self.cond_mlp) else None
        h = []

        for resnet, resnet2, attn, downsample in self.downs:
            x = resnet(x, cond, t)
            x = resnet2(x, cond, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, cond, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, cond, t)
        x = self.pool(x) * 4
        x = self.final_l(x.view(batch, -1))
        # for resnet, resnet2, attn, upsample in self.ups:
        #     x = torch.cat((x, h.pop()), dim=1)
        #     x = resnet(x, cond, t)
        #     x = resnet2(x, cond, t)
        #     x = attn(x)
        #     x = upsample(x)

        return x


class Decoupled_Unet(nn.Module):
    def __init__(
        self,
        dim,
        out_dim = None,
        dim_mults=(1, 2, 4, 8),
        groups = 8,
        channels = 3,
        with_time_emb = True
    ):
        super().__init__()
        self.channels = channels

        dims = [channels, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        if with_time_emb:
            time_dim = dim
            self.time_mlp = nn.Sequential(
                SinusoidalPosEmb(dim),
                nn.Linear(dim, dim * 4),
                Mish(),
                nn.Linear(dim * 4, dim)
            )
        else:
            time_dim = None
            self.time_mlp = None

        self.downs = nn.ModuleList([])
        self.ups1 = nn.ModuleList([])
        self.ups2 = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                ResnetBlock(dim_in, dim_out, time_emb_dim = time_dim),
                ResnetBlock(dim_out, dim_out, time_emb_dim = time_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Downsample(dim_out) if not is_last else nn.Identity()
            ]))

        mid_dim = dims[-1]
        self.mid_block11 = ResnetBlock(mid_dim, mid_dim, time_emb_dim = time_dim)
        self.mid_attn1 = Residual(PreNorm(mid_dim, LinearAttention(mid_dim)))
        self.mid_block12 = ResnetBlock(mid_dim, mid_dim, time_emb_dim = time_dim)

        self.mid_block21 = ResnetBlock(mid_dim, mid_dim, time_emb_dim = time_dim)
        self.mid_attn2 = Residual(PreNorm(mid_dim, LinearAttention(mid_dim)))
        self.mid_block22 = ResnetBlock(mid_dim, mid_dim, time_emb_dim = time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups1.append(nn.ModuleList([
                ResnetBlock(dim_out * 2, dim_in, time_emb_dim = time_dim),
                ResnetBlock(dim_in, dim_in, time_emb_dim = time_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Upsample(dim_in) if not is_last else nn.Identity()
            ]))

            self.ups2.append(nn.ModuleList([
                ResnetBlock(dim_out * 2, dim_in, time_emb_dim = time_dim),
                ResnetBlock(dim_in, dim_in, time_emb_dim = time_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Upsample(dim_in) if not is_last else nn.Identity()
            ]))

        out_dim = default(out_dim, channels)
        self.final_conv1 = nn.Sequential(
            Block(dim, dim),
            nn.Conv2d(dim, out_dim, 1)
        )

        self.final_conv2 = nn.Sequential(
            Block(dim, dim),
            nn.Conv2d(dim, out_dim, 1)
        )

    def forward(self, x, t):
        t = self.time_mlp(t) if exists(self.time_mlp) else None

        h = []

        for resnet, resnet2, attn, downsample in self.downs:
            x = resnet(x, t)
            x = resnet2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        x1 = self.mid_block11(x, t)
        x1 = self.mid_attn1(x1)
        x1 = self.mid_block12(x1, t)

        x2 = self.mid_block21(x, t)
        x2 = self.mid_attn2(x2)
        x2 = self.mid_block12(x2, t)

        num = 1
        for resnet, resnet2, attn, upsample in self.ups1:
            x1 = torch.cat((x1, h[0-num]), dim=1)
            x1 = resnet(x1, t)
            x1 = resnet2(x1, t)
            x1 = attn(x1)
            x1 = upsample(x1)
            num += 1

        x1 = self.final_conv1(x1)
        num = 1
        for resnet, resnet2, attn, upsample in self.ups2:
            x2 = torch.cat((x2, h[0 - num]), dim=1)
            x2 = resnet(x2, t)
            x2 = resnet2(x2, t)
            x2 = attn(x2)
            x2 = upsample(x2)
            num += 1

        x2 = self.final_conv2(x2)

        # 第一个返回为tuple，表示phi的所有参数组合
        return (x1,), x2


if __name__ == "__main__":
    # a = torch.randn(32, 3, 28, 28)
    # time = torch.randint(0, 1000, (32,),device=a.device).long()
    # cond = torch.randint(0, 10, (32,), device=a.device).long()
    # cond_gen = torch.randn(32, 100)
    # md = Cond_Unet(32, channels=3, dim_mults=(1, 2, 4))
    # md = logistic_Unet(16, channels=3, dim_mults=(1, 2, 4))
    # md = Cond_logistic_gen(16, channels=3, dim_mults=(1, 2, 4))
    # # res = md(a, time, cond)
    # loc, log_scale = md(a, time, cond_gen)
    # print(loc.shape, log_scale.shape)

    a = torch.randn(32, 1, 28, 28)
    b = torch.randn(32, 1, 28, 28)
    cond = torch.randn(32, 100)
    time = torch.randint(0, 1000, (32,), device=a.device).long()
    md = Cond_Gen(dim=32, channels=1, dim_mults=(1,2,4))
    # md2 = Dis(dim=32, channels=2, dim_mults=(1,2,4,8))
    res = md(a, time, cond)
    print(res.shape)
    # res2 = md2(a, b, time)
    # print(res2.shape)

    print('*'*80)
    c = torch.randn(32, 1, 28, 28)
    d = torch.randn(32,100)
    label = torch.randint(0,10,(32,)).long()
    time1 = torch.randint(0, 1000, (32,)).long()
    md3 = c_Gen(dim=32, channels=1, dim_mults=(1,2,4))
    res2 = md3(c, time1, d, label)
    print(res2.shape)

    print('*'*80)
    e = torch.randn(32, 1, 28, 28)
    f = torch.randn(32, 1, 28, 28)
    label = torch.randint(0, 10, (32,)).long()
    time1 = torch.randint(0, 1000, (32,)).long()
    md4 = c_Dis(dim=32, channels=2, dim_mults=(1, 2, 4))
    res3 = md4(e, f, time1, label)
    print(res3.shape)


    g = torch.randn(32, 1, 28, 28)
    t = torch.randint(0, 100, (32,)).long()
    md5 = Decoupled_Unet(dim=16, channels=1, dim_mults=(1, 2, 4))
    res4 = md5(g, t)
    print(res4[0][0].shape, res4[1].shape)
