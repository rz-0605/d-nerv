import math
import torch
import torchvision.transforms as transforms
import os
from PIL import Image

def psnr(img1, img2):
    mse = torch.mean((img1/255. - img2/255.) ** 2).item()
    if mse < 1.0e-10:
        return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse)) 

transform = transforms.ToTensor()

images = [["Beauty", 600], ["Bosphorus", 600], ["HoneyBee", 600], ["Jockey", 600], ["ReadySteadyGo", 600], ["ShakeNDry", 300], ["YachtRide", 600]]
patches = 24
patch_dim = (3, 256, 320)
dim = (3, 1024, 1920)
output_base_dir = "./logs/UVG/D-NeRV/Embed1.25_240_256x320_fc_4_5_52_exp2_f8_k3_e400_warm80_b32_lr0.0005_L2_Strd4,2,2,2,2_eval/visualize"
gt_base_dir = "./UVG/gt"

for name, length in images:
    total = 0
    for frame in range(1, length+1):
        gt = torch.empty(dim) 
        output = torch.empty(dim)
        for patch in range(1, patches+1):
            patch_out = transform(Image.open(os.path.join(output_base_dir, f"{name}-{patch:02d}", f"frame{frame:06d}.png")).convert("RGB"))
            patch_gt = transform(Image.open(os.path.join(gt_base_dir, f"{name}-{patch:02d}", f"frame{frame:06d}.png")).convert("RGB"))

            _, patch_h, patch_w = patch_dim
            row = ((patch-1) // 6) * patch_h
            col = ((patch-1) % 6) * patch_w

            output[:, row:row+patch_h, col:col+patch_w] = patch_out
            gt[:, row:row+patch_h, col:col+patch_w] = patch_gt

        total += psnr(gt, output)

    print(f"avg psnr for {name}: {(total / length):04f}")