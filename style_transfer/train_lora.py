# train_lora.py

import os
import glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import torchvision.utils as vutils
from tqdm import tqdm

from model_unet_v2 import UNetSmall
from lora import apply_lora_to_unet


# =====================
# Config
# =====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

STYLE_FOLDER = r"D:\anime_generation\style_images"
BASE_CKPT = r"D:\anime_generation\checkpoints_base_new\model_epoch_049.pth"

OUT_DIR = "checkpoints_lora"
os.makedirs(OUT_DIR, exist_ok=True)

IMG_SIZE = 64
BATCH_SIZE = 4
LR = 5e-5
EPOCHS = 40
DIFFUSION_STEPS = 400


# =====================
# Dataset
# =====================
class StyleDataset(Dataset):
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
        return self.transform(img)


transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(0.5),
    # transforms.ColorJitter(0.05,0.05,0.05,0.01),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
])

dataset = StyleDataset(STYLE_FOLDER, transform)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)


# =====================
# Diffusion Schedule
# =====================
betas = torch.linspace(1e-4, 0.02, DIFFUSION_STEPS, device=DEVICE)
alphas = 1 - betas
alpha_bar = torch.cumprod(alphas, dim=0)
alpha_bar_prev = torch.cat([torch.tensor([1.0], device=DEVICE), alpha_bar[:-1]])


# =====================
# Load base model
# =====================

model = UNetSmall().to(DEVICE)
ckpt = torch.load(BASE_CKPT, map_location=DEVICE)
model.load_state_dict(ckpt["model_state_dict"])
print("Base model loaded.")

RESUME_EPOCH = 35
RESUME_PATH = os.path.join(OUT_DIR, f"lora_epoch_{RESUME_EPOCH}.pth")

if os.path.exists(RESUME_PATH):
    print(f"Resuming LoRA weights from {RESUME_PATH}")
    lora_state = torch.load(RESUME_PATH, map_location=DEVICE)
    # 只加载 LoRA 层（不会加载 base）
    model.load_state_dict(lora_state, strict=False)
    start_epoch = RESUME_EPOCH + 1
else:
    print("No resume checkpoint found.")
    start_epoch = 0



# =====================
# Inject LoRA
# =====================
apply_lora_to_unet(model, r=4, alpha=0.5)

trainable = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(trainable, lr=LR)

print("Trainable params =", sum(p.numel() for p in trainable))


# =====================
# Sample function
# =====================
@torch.no_grad()
def sample(model, shape):
    model.eval()
    x = torch.randn(shape, device=DEVICE)

    for i in reversed(range(DIFFUSION_STEPS)):
        t = torch.full((shape[0],), i, device=DEVICE)
        eps = model(x, t)

        a = alphas[i]
        ab = alpha_bar[i]
        ab_prev_i = alpha_bar_prev[i]
        b = betas[i]

        x0 = (x - torch.sqrt(1 - ab) * eps) / (torch.sqrt(ab) + 1e-8)

        mean = (
            torch.sqrt(ab_prev_i) * b / (1 - ab) * x0 +
            torch.sqrt(a) * (1 - ab_prev_i) / (1 - ab) * x
        )

        if i > 0:
            x = mean + torch.sqrt(b) * torch.randn_like(x)
        else:
            x = mean

    return x.clamp(-1, 1)


# =====================
# Training
# =====================
mse = nn.MSELoss()

print("\nStart LoRA Training...\n")
global_step = 0

for epoch in range(start_epoch, EPOCHS):

    for x0 in tqdm(loader, desc=f"Epoch {epoch}", ncols=80):
        x0 = x0.to(DEVICE)

        t = torch.randint(0, DIFFUSION_STEPS, (x0.size(0),), device=DEVICE)
        noise = torch.randn_like(x0)

        xt = (
            torch.sqrt(alpha_bar[t]).view(-1,1,1,1) * x0 +
            torch.sqrt(1 - alpha_bar[t]).view(-1,1,1,1) * noise
        )

        pred = model(xt, t)
        loss = mse(pred, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        global_step += 1

    # save samples
    samples = sample(model, (8,3,IMG_SIZE,IMG_SIZE))
    samples = samples * 0.5 + 0.5
    vutils.save_image(samples, os.path.join(OUT_DIR, f"sample_epoch_{epoch}.png"))

    # save model
    torch.save(model.state_dict(), os.path.join(OUT_DIR, f"lora_epoch_{epoch}.pth"))

    print(f"Epoch {epoch} saved.")


print("LoRA Finished.")
