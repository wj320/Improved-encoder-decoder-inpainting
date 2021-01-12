import os
import random
import torch
from PIL import Image
from glob import glob


class dataset(torch.utils.data.Dataset):
    def __init__(self, dataset_name, mask_root, img_transform, mask_transform,
                 split='train', target_mask_area=0.1, im_name=False):
        super(dataset, self).__init__()
        self.img_transform = img_transform
        self.mask_transform = mask_transform
        self.target_mask_area = target_mask_area
        self.im_name = im_name

        self.paths = []

        if dataset_name == 'CelebA-HQ':
            if split == 'train':
                image_root = './dataset/CelebA-HQ/train/images'
                files = os.listdir(image_root)
                for file in files:
                    path = os.path.join(image_root, file)
                    self.paths.append(path)
            elif split == 'test':
                image_root = './dataset/CelebA-HQ/test/images'
                files = os.listdir(image_root)
                for file in files:
                    path = os.path.join(image_root, file)
                    self.paths.append(path)

        elif dataset_name == 'imagenet':
            if split == 'train':
                image_list = './dataset/imagenet/train.txt'
            else:
                image_list = './dataset/imagenet/test.txt'
            with open(image_list, 'r') as f:
                files = f.readlines()
            random.shuffle(files)
            for file in files:
                path = file.split()[0]
                self.paths.append(path)

        elif dataset_name == 'places2':
            if split == 'train':
                image_list = './dataset/places2/train.txt'
                with open(image_list, 'r') as f:
                    files = f.readlines()
                random.shuffle(files)
            else:
                image_list = './dataset/places3/val.txt'
                with open(image_list, 'r') as f:
                    files = f.readlines()
            for file in files:
                path = file.split()[0]
                self.paths.append(path)
        elif '.jpg' in dataset_name: # single image
            self.paths.append(dataset_name)

        self.mask_paths = glob('{:s}/*'.format(mask_root))

        self.N_mask = len(self.mask_paths)

    def __getitem__(self, index):
        gt_img = Image.open(self.paths[index])
        gt_img = self.img_transform(gt_img.convert('RGB'))

        mask_path = self.mask_paths[random.randint(0, self.N_mask - 1)]
        mask = Image.open(mask_path)

        mask = self.mask_transform(mask.convert('RGB'))

        im_name = self.paths[index].split('/')[-1]
        if self.im_name:
            return gt_img * mask, mask, gt_img, im_name
        else:
            return gt_img * mask, mask, gt_img

    def __len__(self):
        return len(self.paths)
