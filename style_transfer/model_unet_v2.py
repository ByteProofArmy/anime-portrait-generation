# model_unet.py  —— 与 base diffusion 完全一致的旧结构
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------- 时间嵌入 ----------------
class SinCosPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.post_mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, dim)
        )

    def forward(self, t):
        half = self.dim // 2
        t = t.float() / 1000.0   # 注意：旧版本 base 训练用 1000
        freqs = torch.exp(-math.log(20000) * torch.arange(half, device=t.device).float() / half)
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros(t.size(0), 1, device=t.device)], dim=-1)
        return self.post_mlp(emb)


# ---------------- 基础模块：ResBlock ----------------
class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.gn1 = nn.GroupNorm(8 if out_ch >= 8 else 1, out_ch)
        self.gn2 = nn.GroupNorm(8 if out_ch >= 8 else 1, out_ch)
        self.act = nn.SiLU()

        self.time_proj = nn.Linear(time_emb_dim, out_ch) if time_emb_dim else None
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb=None):
        h = self.act(self.gn1(self.conv1(x)))
        if (self.time_proj is not None) and (t_emb is not None):
            h = h + self.time_proj(t_emb).view(t_emb.size(0), -1, 1, 1)
        h = self.act(self.gn2(self.conv2(h)))
        return h + self.skip(x)


class Downsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.op = nn.Conv2d(ch, ch, 4, 2, 1)

    def forward(self, x):
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


# ---------------- UNet 主体结构（旧版） ----------------
class UNetSmall(nn.Module):
    def __init__(self, in_ch=3, base_ch=64, time_emb_dim=128, channel_mults=(1, 2, 4)):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinCosPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        self.init_conv = nn.Conv2d(in_ch, base_ch, 3, padding=1)

        # Encoder
        self.enc_blocks = nn.ModuleList()
        self.downs = nn.ModuleList()

        ch = base_ch
        for m in channel_mults:
            out_ch = base_ch * m
            self.enc_blocks.append(ResidualBlock(ch, out_ch, time_emb_dim))
            self.downs.append(Downsample(out_ch))
            ch = out_ch

        # bottleneck
        self.mid1 = ResidualBlock(ch, ch * 2, time_emb_dim)
        self.mid2 = ResidualBlock(ch * 2, ch, time_emb_dim)

        # Decoder
        self.ups = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()

        for m in reversed(channel_mults):
            out_ch = base_ch * m
            self.ups.append(Upsample(ch))
            self.dec_blocks.append(ResidualBlock(ch + out_ch, out_ch, time_emb_dim))
            ch = out_ch

        self.final_conv = nn.Sequential(
            nn.GroupNorm(8 if ch >= 8 else 1, ch),
            nn.SiLU(),
            nn.Conv2d(ch, in_ch, 3, padding=1),
        )

    def forward(self, x, t):
        t_emb = self.time_mlp(t)

        h = self.init_conv(x)
        skips = []

        # Encoder
        for block, down in zip(self.enc_blocks, self.downs):
            h = block(h, t_emb)
            skips.append(h)
            h = down(h)

        # bottleneck
        h = self.mid1(h, t_emb)
        h = self.mid2(h, t_emb)

        # Decoder
        for up, block in zip(self.ups, self.dec_blocks):
            h = up(h)
            skip = skips.pop()
            if skip.shape[-2:] != h.shape[-2:]:
                skip = skip[..., :h.shape[-2], :h.shape[-1]]
            h = torch.cat([h, skip], dim=1)
            h = block(h, t_emb)

        return self.final_conv(h)
