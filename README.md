# Improved-encoder-decoder-based-network-for-image-inpainting

## Installation
Clone this repo.
```
git clone https://github.com/wj320/Improved-encoder-decoder-inpainting.git
cd  Improved-encoder-decoder-inpainting/
```

## Requirements
* Python 3.6+
* Pytorch 0.4.1+

## Dataset
* CelebA-HQ: http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
* ImageNet: http://www.image-net.org/
* Places2: http://places2.csail.mit.edu/download.html

## Mask:
* qd_imd: https://github.com/karfly/qd-imd

## Train
```
CUDA_VISIBLE_DEVICES=<gpu_id> python train.py  --dataset_name='imagenet' --batch_size=8  --image_size=256 --mask_root='./download_mask/qd_imd/train/' --save_dir='./snapshot/imagenet/'
```

## Test
```
python test_batch.py  --image_size=256  --mask_root='./masks/download_mask/qd_imd/test/'   --snapshot='./snapshot/imagenet/ckpt/300000.pth'  --save_path='test_result/imagenet/'
```
