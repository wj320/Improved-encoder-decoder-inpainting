import argparse
from torchvision import transforms
from model import UNet
from datasets import dataset
from evaluation import *
from util.io import load_ckpt

parser = argparse.ArgumentParser()
# training options
parser.add_argument('--dataset_name', type=str, default='imagenet')
parser.add_argument('--mask_root', type=str, default='./qd_imd/test/')
parser.add_argument('--snapshot', type=str, default='./snapshot/imagenet/ckpt/300000.pth')
parser.add_argument('--image_size', type=int, default=256)
parser.add_argument('--save_path', type=str, default='result/imagenet/')
args = parser.parse_args()


device = torch.device('cuda')

size = (args.image_size, args.image_size)
img_transform = transforms.Compose(
    [transforms.Resize(size=size), transforms.ToTensor(),
     transforms.Normalize(mean=opt.MEAN, std=opt.STD)])
mask_transform = transforms.Compose(
    [transforms.Resize(size=size), transforms.ToTensor()])

dataset_val = dataset(args.dataset_name, args.mask_root, img_transform, mask_transform, 'test', im_name=True)

model = UNet().to(device)
load_ckpt(args.snapshot, [('model', model)])

model.eval()

#evaluate(model, dataset_val, device, 'result.jpg')
save = True
save_path = args.save_path
if save:
    if not os.path.exists(save_path):
        os.makedirs(save_path)

evaluate_psnr_ssim(model, dataset_val, device, save, save_path)

torch.cuda.empty_cache()