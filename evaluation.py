import os
import torch
import torch.nn as nn
from torchvision.utils import make_grid
from torchvision.utils import save_image
from util.image import *
from util.io import save_tensor_image

def evaluate_l1_l2(model, dataset, device):
    l1_loss = nn.L1Loss(reduction='sum')
    l2_loss = nn.MSELoss(reduction='sum')
    out_l1_loss = []
    out_l2_loss = []
    comp_l1_loss = []
    comp_l2_loss = []

    for i in range(0, len(dataset), 8):
        image, mask, gt = zip(*[dataset[j] for j in range(i, i+8)])
        image = torch.stack(image)
        mask = torch.stack(mask)
        gt = torch.stack(gt)
        with torch.no_grad():
            output, _ = model(image.to(device), mask.to(device))
        output = output.to(torch.device('cpu'))
        output_comp = mask * image + (1 - mask) * output

        out_l1_loss.append(l1_loss(output, gt))
        out_l2_loss.append(l2_loss(output, gt))

        comp_l1_loss.append(l1_loss(output_comp, gt))
        comp_l2_loss.append(l2_loss(output_comp, gt))


    out_l1_loss_avg = sum(out_l1_loss) / len(out_l1_loss)
    comp_l1_loss_avg = sum(comp_l1_loss) / len(comp_l1_loss)
    out_l2_loss_avg = sum(out_l2_loss) / len(out_l2_loss)
    comp_l2_loss_avg = sum(comp_l2_loss) / len(comp_l2_loss)
    print('l1 loss: output: %.3f  comp: %.3f'.format(out_l1_loss_avg, comp_l1_loss_avg))
    print('l2 loss: output: %.3f  comp: %.3f'.format(out_l2_loss_avg, comp_l2_loss_avg))

def evaluate_psnr_ssim(model, dataset, device, save, save_path):

    out_psnr = []
    out_ssim = []
    comp_psnr = []
    comp_ssim = []

    for i in range(0, len(dataset), 8):
        image, mask, gt, im_name = zip(*[dataset[j] for j in range(i, i+8)])
        image = torch.stack(image)
        mask = torch.stack(mask)
        gt = torch.stack(gt)
        with torch.no_grad():
            output, _ = model(image.to(device), mask.to(device))
        output = output.to(torch.device('cpu'))
        output_comp = mask * image + (1 - mask) * output

        out_psnr.append(psnr(output, gt))
        out_ssim.append(ssim(output, gt))

        comp_psnr.append(psnr(output_comp, gt))
        comp_ssim.append(ssim(output_comp, gt))



        if save:
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            image = unnormalize(image)
            gt = unnormalize(gt)
            output = unnormalize(output)
            output_comp = unnormalize(output_comp)

            for j in range(8):
                save_im_name = im_name[j].split('.')[0] + '.png'
                save_im_name = os.path.join(save_path, save_im_name)
                images = torch.stack((image[j], mask[j], gt[j],
                           output[j], output_comp[j]))
                grid = make_grid(images)

                save_image(grid, save_im_name)

            '''
            output = unnormalize(output)
            output_comp = unnormalize(output_comp)
            for j in range(8):
                save_im_name = im_name[j].split('.')[0]
                save_tensor_image(output[j], filename=os.path.join(save_path, save_im_name+'_out.png'))
                save_tensor_image(output_comp[j], filename=os.path.join(save_path, save_im_name+'_comp.png'))
            '''

    print('PSNR: output: %.3f  comp: %.3f' % (sum(out_psnr)/len(out_psnr), sum(comp_psnr)/len(comp_psnr)))
    print('SSIM: output: %.3f  comp: %.3f' % (sum(out_ssim)/len(out_ssim), sum(comp_ssim)/len(comp_ssim)))

def evaluate(model, dataset, device, filename):
    image, mask, gt = zip(*[dataset[i] for i in range(8)])
    image = torch.stack(image)
    mask = torch.stack(mask)
    gt = torch.stack(gt)
    with torch.no_grad():
        output, _ = model(image.to(device), mask.to(device))
    output = output.to(torch.device('cpu'))
    output_comp = mask * image + (1 - mask) * output

    grid = make_grid(
        torch.cat((unnormalize(image), mask, unnormalize(output),
                   unnormalize(output_comp), unnormalize(gt)), dim=0))
    save_image(grid, filename)

def evaluate_single(model, dataset, device, filename):
    image, mask, gt = dataset[0]
    image = torch.unsqueeze(image, dim=0)
    mask = torch.unsqueeze(mask, dim=0)
    gt = torch.unsqueeze(gt, dim=0)
    with torch.no_grad():
        output, _ = model(image.to(device), mask.to(device))
    output = output.to(torch.device('cpu'))
    output_comp = mask * image + (1 - mask) * output

    grid = make_grid(
        torch.cat((unnormalize(image), mask,
                   unnormalize(output),
                   unnormalize(output_comp),
                   unnormalize(gt)), dim=0))
    save_image(grid, filename)

def evaluate_AbnNet(model, dataset, device, save, save_path, mode):

    out_psnr = []
    out_ssim = []
    comp_psnr = []
    comp_ssim = []

    for i in range(0, len(dataset)):
        image, mask, gt, im_name = dataset[i]
        image = torch.unsqueeze(image, 0)
        mask = torch.unsqueeze(mask, 0)
        feature_save_path = os.path.join(save_path, im_name.split('.')[0])
        with torch.no_grad():
            if mode == 'save':
                model(image.to(device), mask.to(device), mode, feature_save_path)
            else:
                output, _ = model(image.to(device), mask.to(device), mode, feature_save_path)
                output = output.to(torch.device('cpu'))
                output_comp = mask * image + (1 - mask) * output


                if save:
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)

                    image = unnormalize(image)
                    output = unnormalize(output)
                    output_comp = unnormalize(output_comp)


                    save_im_name = im_name.split('.')[0] + '.png'
                    save_im_name = os.path.join(save_path, save_im_name)
                    images = torch.stack((image[0], mask[0], gt,
                            output[0], output_comp[0]))
                    grid = make_grid(images)

                    save_image(grid, save_im_name)

