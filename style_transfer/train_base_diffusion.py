import os
import glob
import math
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as transforms
import torchvision.utils as vutils
from ema import EMA


# ------------------ 配置 ------------------
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_FOLDER = r"D:\anime_generation\images"
IMG_SIZE = 64
BATCH_SIZE = 4
NUM_WORKERS = 0
DIFFUSION_STEPS = 200

EPOCHS = 50               # 总 epoch
MAX_BATCHES = 200
LR = 1e-4

SAMPLE_BATCH = 8
OUT_DIR = "./checkpoints_base_new"
os.makedirs(OUT_DIR, exist_ok=True)

print("Using device:", DEVICE)
print("Data folder:", DATA_FOLDER)

# ------------------ 数据集 ------------------
class ReadingImages(Dataset):
    def __init__(self, folder, transform=None):
        self.files = sorted(
            glob.glob(os.path.join(folder, "*.png")) +
            glob.glob(os.path.join(folder, "*.jpg"))
        )
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2,0.2,0.2,0.08),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
])

dataset = ReadingImages(DATA_FOLDER, transform)
loader = DataLoader(dataset, batch_size=BATCH_SIZE,
                    shuffle=True, num_workers=NUM_WORKERS, drop_last=True)

print("Dataset size:", len(dataset))

# ------------------ Diffusion schedule ------------------
betas = torch.linspace(1e-4, 0.02, DIFFUSION_STEPS, device=DEVICE)
alphas = 1.0 - betas
cum_alpha_bar = torch.cumprod(alphas, dim=0)
alpha_bar_prev = torch.cat([torch.tensor([1.0], device=DEVICE), cum_alpha_bar[:-1]])

# ------------------ 时间嵌入 ------------------
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
        t = t.float() / float(DIFFUSION_STEPS)
        freqs = torch.exp(-math.log(20000) * torch.arange(half, device=t.device).float()/half)
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros(t.size(0), 1, device=t.device)], dim=-1)
        return self.post_mlp(emb)

# ------------------ UNet ------------------
class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.gn1 = nn.GroupNorm(8 if out_ch>=8 else 1, out_ch)
        self.gn2 = nn.GroupNorm(8 if out_ch>=8 else 1, out_ch)
        self.act = nn.SiLU()
        self.time_proj = nn.Linear(time_emb_dim, out_ch) if time_emb_dim else None
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb=None):
        h = self.act(self.gn1(self.conv1(x)))
        if t_emb is not None:
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
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)

class UNetSmall(nn.Module):
    def __init__(self, in_ch=3, base_ch=64, time_emb_dim=128, channel_mults=(1,2,4)):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinCosPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        self.init_conv = nn.Conv2d(in_ch, base_ch, 3, padding=1)
        chs = [base_ch*m for m in channel_mults]
        self.enc_blocks = nn.ModuleList()
        self.downs = nn.ModuleList()
        in_c = base_ch
        for out_c in chs:
            self.enc_blocks.append(ResidualBlock(in_c, out_c, time_emb_dim))
            self.downs.append(Downsample(out_c))
            in_c = out_c

        self.mid1 = ResidualBlock(in_c, in_c*2, time_emb_dim)
        self.mid2 = ResidualBlock(in_c*2, in_c, time_emb_dim)

        self.ups = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        for out_c in reversed(chs):
            self.ups.append(Upsample(in_c))
            self.dec_blocks.append(ResidualBlock(in_c + out_c, out_c, time_emb_dim))
            in_c = out_c

        self.final_conv = nn.Sequential(
            nn.GroupNorm(8 if in_c>=8 else 1, in_c),
            nn.SiLU(),
            nn.Conv2d(in_c, in_ch, 3, padding=1)
        )

    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        h = self.init_conv(x)
        skips = []
        for enc,down in zip(self.enc_blocks,self.downs):
            h = enc(h,t_emb)
            skips.append(h)
            h = down(h)
        h = self.mid1(h,t_emb)
        h = self.mid2(h,t_emb)
        for up,dec,skip in zip(self.ups,self.dec_blocks,reversed(skips)):
            h = up(h)
            if skip.shape[-1]!=h.shape[-1]:
                skip = skip[..., :h.shape[-2], :h.shape[-1]]
            h = torch.cat([h, skip], dim=1)
            h = dec(h,t_emb)
        return self.final_conv(h)

model = UNetSmall().to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=LR)

ema = EMA(model, decay=0.995)
mse = nn.MSELoss()

# ------------------ Resume Training ------------------
resume_list = sorted([f for f in os.listdir(OUT_DIR) if f.startswith("model_epoch_")])
start_epoch = 0 

if resume_list:
    last_ckpt = os.path.join(OUT_DIR, resume_list[-1])
    print(f"🔄 Resuming from: {last_ckpt}")
    ckpt = torch.load(last_ckpt, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    opt.load_state_dict(ckpt["optimizer_state_dict"])

    if "ema_state" in ckpt:
        ema.shadow = {k: v.clone() for k, v in ckpt["ema_state"].items()}


    start_epoch = ckpt["epoch"] + 1
else:
    print("🆕 No checkpoint found. Training from scratch.")

# ------------------ Sampling ------------------
@torch.no_grad()
def sample(model, ema, shape):
    # 使用 EMA 权重
    ema.apply_shadow(model)

    model.eval()
    x = torch.randn(shape, device=DEVICE)

    for step in reversed(range(DIFFUSION_STEPS)):
        t = torch.full((shape[0],), step, device=DEVICE, dtype=torch.long)
        eps = model(x, t)

        a = alphas[step]
        ab = cum_alpha_bar[step]
        ab_prev = alpha_bar_prev[step]
        b = betas[step]

        x0 = (x - torch.sqrt(1 - ab) * eps) / (torch.sqrt(ab) + 1e-8)

        mean = (
            torch.sqrt(ab_prev) * b / (1 - ab) * x0 +
            torch.sqrt(a) * (1 - ab_prev) / (1 - ab) * x
        )

        if step > 0:
            x = mean + torch.sqrt(b) * torch.randn_like(x)
        else:
            x = mean

    # 还原模型参数，否则训练会崩
    ema.restore(model)

    return x.clamp(-1, 1)


# ------------------ Training Loop ------------------
print("\n🚀 Starting training...")

for epoch in range(start_epoch, EPOCHS):

    for batch_idx, x0 in enumerate(loader):
        if batch_idx >= MAX_BATCHES:
            break

        x0 = x0.to(DEVICE)
        t = torch.randint(0, DIFFUSION_STEPS, (x0.size(0),), device=DEVICE)
        noise = torch.randn_like(x0)

        xt = (
            torch.sqrt(cum_alpha_bar[t]).view(-1,1,1,1) * x0 +
            torch.sqrt(1-cum_alpha_bar[t]).view(-1,1,1,1) * noise
        )

        pred = model(xt, t)
        loss = mse(pred, noise)

        opt.zero_grad()
        loss.backward()
        opt.step()

        ema.update(model)

        if batch_idx % 20 == 0:
            print(f"Epoch {epoch:03d} Batch {batch_idx:04d} Loss: {loss.item():.6f}")

    # -------- Save sample --------
    samples = sample(model, ema, (SAMPLE_BATCH, 3, IMG_SIZE, IMG_SIZE))

    samples_01 = samples * 0.5 + 0.5
    grid_path = os.path.join(OUT_DIR, f"sample_epoch_{epoch:03d}.png")
    vutils.save_image(vutils.make_grid(samples_01, nrow=4), grid_path)
    print(f"🖼 Saved sample: {grid_path}")

    # -------- Save checkpoint --------
    ckpt_path = os.path.join(OUT_DIR, f"model_epoch_{epoch:03d}.pth")
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": opt.state_dict(),
        "ema_state": ema.shadow
    }, ckpt_path)
    print(f"💾 Saved checkpoint: {ckpt_path}")

print("\n🎉 Training finished!")
