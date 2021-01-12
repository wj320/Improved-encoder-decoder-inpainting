import argparse
import numpy as np
import os
import torch
from torch.utils import data
from torchvision import transforms
import opt
from evaluation import *
from loss import InpaintingLoss
from model import VGG16FeatureExtractor, UNet
from datasets import dataset
from util.io import load_ckpt
from util.io import save_ckpt


class InfiniteSampler(data.sampler.Sampler):
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __iter__(self):
        return iter(self.loop())

    def __len__(self):
        return 2 ** 31

    def loop(self):
        i = 0
        order = np.random.permutation(self.num_samples)
        while True:
            yield order[i]
            i += 1
            if i >= self.num_samples:
                np.random.seed()
                order = np.random.permutation(self.num_samples)
                i = 0


parser = argparse.ArgumentParser()
# training options
parser.add_argument('--dataset_name', type=str, default='imagenet')
parser.add_argument('--mask_root', type=str, default='./qd_imd/train/')
parser.add_argument('--save_dir', type=str, default='./snapshot/imagenet/')
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--lr_finetune', type=float, default=5e-5)
parser.add_argument('--max_iter', type=int, default=300000)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--n_threads', type=int, default=16)
parser.add_argument('--save_model_interval', type=int, default=50000)
parser.add_argument('--vis_interval', type=int, default=5000)
parser.add_argument('--log_interval', type=int, default=100)
parser.add_argument('--image_size', type=int, default=256)
parser.add_argument('--resume', type=str, default='')
parser.add_argument('--finetune', action='store_true')

args = parser.parse_args()

torch.backends.cudnn.benchmark = True
device = torch.device('cuda')


if not os.path.exists(args.save_dir+'/images'):
    os.makedirs('{:s}/images'.format(args.save_dir))
    os.makedirs('{:s}/ckpt'.format(args.save_dir))


size = (args.image_size, args.image_size)
img_tf = transforms.Compose(
    [transforms.Resize(size=size), transforms.ToTensor(),
     transforms.Normalize(mean=opt.MEAN, std=opt.STD)])
mask_tf = transforms.Compose(
    [transforms.Resize(size=size), transforms.ToTensor()])

dataset_train = dataset(args.dataset_name, args.mask_root, img_tf, mask_tf, 'train', im_name=False)
dataset_val = dataset(args.dataset_name, args.mask_root, img_tf, mask_tf, 'val', im_name=True)

iterator_train = iter(data.DataLoader(
    dataset_train, batch_size=args.batch_size,
    sampler=InfiniteSampler(len(dataset_train)),
    num_workers=args.n_threads, drop_last=True))

model = UNet().to(device)


if args.finetune:
    lr = args.lr_finetune
    model.freeze_enc_bn = True
else:
    lr = args.lr

start_iter = 0
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
criterion = InpaintingLoss(VGG16FeatureExtractor()).to(device)

if args.resume:
    start_iter = load_ckpt(
        args.resume, [('model', model)], [('optimizer', optimizer)])
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print('Starting from iter ', start_iter)


for i in range(start_iter, args.max_iter):
    model.train()
    image, mask, gt = [x.to(device) for x in next(iterator_train)]

    output, _ = model(image, mask)
    loss_dict = criterion(image, mask, output, gt)

    loss = 0.0
    printer = 'iter: {}  '.format(i+1)
    for key, coef in opt.LAMBDA_DICT.items():
        value = coef * loss_dict[key]
        loss += value
        printer += 'loss_{:s}  {:f}  '.format(key, value.item())
    if (i + 1) % args.log_interval == 0:
        print(printer)


    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (i + 1) % args.save_model_interval == 0 or (i + 1) == args.max_iter:
        save_ckpt('{:s}/ckpt/{:d}.pth'.format(args.save_dir, i + 1),
                  [('model', model)], [('optimizer', optimizer)], i + 1)

    if (i + 1) % args.vis_interval == 0:
        model.eval()
        save = True
        save_path = os.path.join(args.save_dir, 'images')
        evaluate_psnr_ssim(model, dataset_val, device, save, save_path)

torch.cuda.empty_cache()
