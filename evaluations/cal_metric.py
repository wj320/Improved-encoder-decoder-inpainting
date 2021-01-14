import numpy as np
from PIL import Image
import os
import math
from skimage.metrics import structural_similarity as ssim


def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))



gt_root = './dataset/CelebA-HQ/test/images/'
out_root = './images/'

outputs = os.listdir(out_root)

out_psnr = []
out_ssim = []
out_norm = []

for name in outputs:

    out_path = os.path.join(out_root, name)
    out = Image.open(out_path).convert('RGB')
    out = np.array(out).astype(np.float)

    name = name.split('.')[0] + '.jpg'
    gt_path = os.path.join(gt_root, name)
    gt = Image.open(gt_path).convert('RGB')
    gt = gt.resize((256, 256))
    gt = np.array(gt).astype(np.float)

    s = np.size(gt)
    norm1 = np.linalg.norm((out.reshape(-1) / 255 - gt.reshape(-1) / 255), ord=1, axis=None, keepdims=False) / s

    out_psnr.append(psnr(out, gt))
    out_ssim.append(ssim(out, gt, multichannel=True))
    out_norm.append(norm1)


print('mean PSNR: ', sum(out_psnr)/len(out_psnr))
print('mean SSIM: ', sum(out_ssim)/len(out_ssim))
print('mean l1: ', sum(out_norm)/len(out_norm))