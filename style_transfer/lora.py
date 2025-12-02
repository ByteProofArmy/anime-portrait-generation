# lora.py
import torch
import torch.nn as nn


class LoRA(nn.Module):
    def __init__(self, module: nn.Conv2d, r=4, alpha=1.0):
        super().__init__()
        self.module = module
        self.r = r

        in_ch = module.in_channels
        out_ch = module.out_channels
        k = module.kernel_size
        stride = module.stride
        padding = module.padding
        bias = module.bias is not None

        # 冻结原始权重
        for p in module.parameters():
            p.requires_grad = False

        # LoRA branch
        self.lora_down = nn.Conv2d(in_ch, r, kernel_size=1, bias=False)
        self.lora_up = nn.Conv2d(r, out_ch, kernel_size=k,
                                 stride=stride, padding=padding,
                                 bias=bias)
        nn.init.zeros_(self.lora_up.weight)
        self.scale = alpha / r

    def forward(self, x):
        return self.module(x) + self.scale * self.lora_up(self.lora_down(x))


def apply_lora_to_unet(model, r=4, alpha=1.0):
    """
    在 UNetSmall 中找到所有 Conv2d 层并替换成带 LoRA 的卷积
    """
    count = 0
    for name, module in model.named_modules():

        # 只给 3×3 或 1×1 的 conv 加 LoRA
        if isinstance(module, nn.Conv2d):
            if module.kernel_size in [(3, 3), (1, 1)]:
                parent = get_parent_module(model, name)
                attr = name.split(".")[-1]

                setattr(parent, attr, LoRA(module, r=r, alpha=alpha))
                count += 1

    print(f"✨ Injected LoRA into {count} Conv2d layers.")
    return model


def get_parent_module(root, name: str):
    parts = name.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent
