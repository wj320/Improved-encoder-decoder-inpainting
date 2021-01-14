import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data
import torchvision.transforms as transforms

from torchvision.models.inception import inception_v3
import numpy as np
import math
from PIL import Image
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--image_size', type=int, default=256)
parser.add_argument('--image_root', type=str, default='./test_result/Pconv/CelebA-HQ/256/ratio_3_scale_60_out/')
parser.add_argument('--num_test', type=int, default=2000,
                    help='how many images to load for each test')
args = parser.parse_args()

def get_inception_score(imgs, batch_size=32, resize=True, splits=10):
    """Computes the inception score of the generated images imgs
    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    N = imgs.shape[0]

    #assert batch_size > 0
    #assert N > batch_size

    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=True)
    #up = nn.Upsample(size=(299, 299), mode='bilinear', align_corners=True)
    up = nn.Upsample(size=(256, 256), mode='bilinear', align_corners=True)
    if torch.cuda.is_available():
        inception_model.cuda(0)
        up.cuda(0)
    inception_model.eval()

    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x, dim=-1).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, 1000))
    n_batches = int(math.ceil(float(N) / float(batch_size)))

    for i in range(n_batches):
        batch = torch.from_numpy(imgs[i * batch_size:min((i + 1) * batch_size, N)])
        batchv = Variable(batch)
        if torch.cuda.is_available():
            batchv = batchv.cuda(0)

        preds[i * batch_size:min((i + 1) * batch_size, N)] = get_pred(batchv)

    # Now compute the mean kl-div
    scores = []

    for i in range(splits):
        part = preds[(i * preds.shape[0] // splits):((i + 1) * preds.shape[0] // splits), :]
        kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
        kl = np.mean(np.sum(kl, 1))
        scores.append(np.exp(kl))

    return np.mean(scores), np.std(scores)


if __name__=='__main__':


    size = (args.image_size, args.image_size)
    #img_transform = transforms.Compose(
    #    [transforms.Resize(size=size), transforms.ToTensor(),
    #     transforms.Normalize(mean=opt.MEAN, std=opt.STD)])

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])

    images = os.listdir(args.image_root)
    iters = int(len(images) / args.num_test)

    u = np.zeros(iters, np.float32)
    sigma = np.zeros(iters, np.float32)

    for i in range(iters):
        all_samples = []
        num = i*args.num_test

        for j in range(args.num_test):
            index = num+j
            image_path = os.path.join(args.image_root, images[index])
            image = Image.open(image_path)
            image = transform(image.convert('RGB'))
            all_samples.append(image)

        all_samples = np.stack(all_samples)
        u_iter, sigma_iter = get_inception_score(all_samples, batch_size=8, resize=False)

        u[i] = u_iter
        sigma[i] = sigma_iter

        print(i)
        print('{:10.4f},{:10.4f}'.format(u_iter, sigma_iter))


    print('{:>10},{:>10}'.format('u', 'sigma'))
    print('{:10.4f},{:10.4f}'.format(u.mean(), sigma.mean()))