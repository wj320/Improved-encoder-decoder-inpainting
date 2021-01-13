import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from sync_batchnorm.batchnorm import SynchronizedBatchNorm2d


def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find(
                'Linear') == 0) and hasattr(m, 'weight'):
            if init_type == 'gaussian':
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    return init_fun


class VGG16FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        vgg16 = models.vgg16(pretrained=True)
        self.enc_1 = nn.Sequential(*vgg16.features[:5])
        self.enc_2 = nn.Sequential(*vgg16.features[5:10])
        self.enc_3 = nn.Sequential(*vgg16.features[10:17])

        # fix the encoder
        for i in range(3):
            for param in getattr(self, 'enc_{:d}'.format(i + 1)).parameters():
                param.requires_grad = False

    def forward(self, image):
        results = [image]
        for i in range(3):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

# PartialConv: Image Inpainting for Irregular Holes Using Partial Convolutions [Liu+, arXiv2018].
# The code is from https://github.com/naoto0804/pytorch-inpainting-with-partial-conv
class PartialConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super().__init__()
        self.input_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                    stride, padding, dilation, groups, bias)
        self.mask_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                   stride, padding, dilation, groups, False)
        self.input_conv.apply(weights_init('kaiming'))

        torch.nn.init.constant_(self.mask_conv.weight, 1.0)

        # mask is not updated
        for param in self.mask_conv.parameters():
            param.requires_grad = False

    def forward(self, input, mask):
        # http://masc.cs.gmu.edu/wiki/partialconv
        # C(X) = W^T * X + b, C(0) = b, D(M) = 1 * M + 0 = sum(M)
        # W^T* (M .* X) / sum(M) + b = [C(M .* X) – C(0)] / D(M) + C(0)

        output = self.input_conv(input * mask)
        if self.input_conv.bias is not None:
            output_bias = self.input_conv.bias.view(1, -1, 1, 1).expand_as(
                output)
        else:
            output_bias = torch.zeros_like(output)

        with torch.no_grad():
            output_mask = self.mask_conv(mask)

        no_update_holes = output_mask == 0
        mask_sum = output_mask.masked_fill_(no_update_holes, 1.0)

        output_pre = (output - output_bias) / mask_sum + output_bias
        output = output_pre.masked_fill_(no_update_holes, 0.0)

        new_mask = torch.ones_like(output)
        new_mask = new_mask.masked_fill_(no_update_holes, 0.0)

        return output, new_mask

class _NonLocalBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels, out_channels, dimension=2, sub_sample=True,
                 W_bn_layer=True, sample='none-3', out_bn=True, out_activ='leaky', conv_bias=False):
        super(_NonLocalBlockND, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample
        self.W_bn_layer = W_bn_layer
        self.out_bn = out_bn

        self.in_channels = in_channels
        self.inter_channels = inter_channels
        self.out_channels = out_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            self.pool = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.InstanceNorm3d
        elif dimension == 2:
            self.pool = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.InstanceNorm2d
        else:
            self.pool = nn.MaxPool1d(kernel_size=(2))
            bn = nn.InstanceNorm1d

        pconv = PartialConv

        self.g = pconv(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        self.theta = pconv(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        self.phi = pconv(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        self.W = pconv(in_channels=self.inter_channels, out_channels=self.in_channels,
                       kernel_size=1, stride=1, padding=0)
        if W_bn_layer:
            self.W_bn = bn(self.in_channels)


        if sample == 'down-5':
            self.pconv = pconv(in_channels=self.in_channels, out_channels=self.out_channels,
                          kernel_size=5, stride=2, padding=2, bias=conv_bias)
        elif sample == 'down-7':
            self.pconv = pconv(in_channels=self.in_channels, out_channels=self.out_channels,
                          kernel_size=7, stride=2, padding=3, bias=conv_bias)
        elif sample == 'down-3':
            self.pconv = pconv(in_channels=self.in_channels, out_channels=self.out_channels,
                          kernel_size=3, stride=2, padding=1, bias=conv_bias)
        else:
            self.pconv = pconv(in_channels=self.in_channels, out_channels=self.out_channels,
                          kernel_size=3, stride=1, padding=1, bias=conv_bias)

        if self.out_bn:
            self.bn = bn(out_channels)
        if out_activ == 'relu':
            self.activation = nn.ReLU()
        elif out_activ == 'leaky':
            self.activation = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x, mask, return_nl_map=False):
        """
        :param x: (b, c, t, h, w)
        :param return_nl_map: if True return z, nl_map, else only return z.
        :return:
        """

        batch_size = x.size(0)

        theta_x, theta_mask = self.theta(x, mask)

        phi_x, phi_mask = self.phi(x, mask)

        if self.sub_sample:
            phi_x = self.pool(phi_x)
            phi_mask = self.pool(phi_mask)

        f = torch.matmul(theta_x.view(batch_size, self.inter_channels, -1).permute(0, 2, 1),
                         phi_x.view(batch_size, self.inter_channels, -1))

        f_mask = torch.matmul(theta_mask.view(batch_size, self.inter_channels, -1).permute(0, 2, 1),
                              phi_mask.view(batch_size, self.inter_channels, -1))

        N = f.size(-1)
        f_div_C = f * f_mask / N

        g_x, g_mask = self.g(x, mask)

        if self.sub_sample:
            g_x = self.pool(g_x)
            g_mask = self.pool(g_mask)


        y_mask = torch.matmul(f_mask, g_mask.view(batch_size, self.inter_channels, -1).
                         permute(0, 2, 1)).permute(0, 2, 1). contiguous().\
                        view(batch_size, self.inter_channels, *x.size()[2:])

        y = torch.matmul(f_div_C, g_x.view(batch_size, self.inter_channels, -1).
                         permute(0, 2, 1)).permute(0, 2, 1). contiguous().\
                        view(batch_size, self.inter_channels, *x.size()[2:])  * y_mask

        W_y, Wy_mask = self.W(y, y_mask)
        if self.W_bn_layer:
            W_y = self.W_bn(W_y)

        z, new_mask = self.pconv(W_y + x, torch.logical_or(mask, Wy_mask).float())

        if return_nl_map:
            return self.activation(z), f_div_C

        if self.out_bn:
            return self.activation(self.bn(z)), new_mask
        else:
            return self.activation(z), new_mask


# Implementation of SAP
class NONLocalBlock2D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels, out_channels, sub_sample=True, W_bn_layer=True,
                 sample='none-3', out_bn=True, out_activ='relu',):
        super(NONLocalBlock2D, self).__init__(in_channels, inter_channels, out_channels,
                                              dimension=2, sub_sample=sub_sample, W_bn_layer=W_bn_layer,
                                              sample=sample, out_bn=out_bn, out_activ=out_activ,)


class Convblock(nn.Module):
    def __init__(self, in_ch, out_ch, bn=True, sample='none-3', activ='leaky',
                 conv_bias=False):
        super().__init__()
        if sample == 'down-5':
            self.conv = PartialConv(in_ch, out_ch, 5, 2, 2, bias=conv_bias)
        elif sample == 'down-7':
            self.conv = PartialConv(in_ch, out_ch, 7, 2, 3, bias=conv_bias)
        elif sample == 'down-3':
            self.conv = PartialConv(in_ch, out_ch, 3, 2, 1, bias=conv_bias)
        else:
            self.conv = PartialConv(in_ch, out_ch, 3, 1, 1, bias=conv_bias)

        if bn:
            #self.bn = nn.BatchNorm2d(out_ch)
            self.bn = nn.InstanceNorm2d(out_ch)
        if activ == 'relu':
            self.activation = nn.ReLU()
        elif activ == 'leaky':
            self.activation = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, input, input_mask):
        h, h_mask = self.conv(input, input_mask)
        if hasattr(self, 'bn'):
            h = self.bn(h)
        if hasattr(self, 'activation'):
            h = self.activation(h)
        return h, h_mask

class MFT(nn.Module):
    def __init__(self, in_ch1, in_ch2, out_ch, bn=True, activ='leaky',
                 conv_bias=False, param_free_norm_type='syncbatch'):
        super().__init__()

        self.conv_h1 = PartialConv(in_ch1, out_ch, 3, 1, 1, bias=conv_bias)
        self.conv_h2 = PartialConv(in_ch2, out_ch, 3, 1, 1, bias=conv_bias)
        self.conv_concate = PartialConv(in_ch1+in_ch2, out_ch, 3, 1, 1, bias=conv_bias)


        if param_free_norm_type == 'instance':
            self.param_free_norm = nn.InstanceNorm2d(out_ch, affine=False)
        elif param_free_norm_type == 'syncbatch':
            self.param_free_norm = SynchronizedBatchNorm2d(out_ch, affine=False)
        elif param_free_norm_type == 'batch':
            self.param_free_norm = nn.BatchNorm2d(out_ch, affine=False)
        else:
            raise ValueError('%s is not a recognized param-free norm type in SPADE'
                             % param_free_norm_type)

        if bn:
            self.pre_conv_h1 = nn.Sequential(nn.Conv2d(in_ch1, 1, kernel_size=1, padding=0), nn.ReLU())
            self.pre_conv_h2 = nn.Sequential(nn.Conv2d(in_ch2, 1, kernel_size=1, padding=0), nn.ReLU())

            self.conv_gamma1 = nn.Conv2d(1, in_ch1, kernel_size=1, padding=0)
            self.conv_beta1 = nn.Conv2d(1, in_ch1, kernel_size=1, padding=0)

            self.conv_gamma2 = nn.Conv2d(1, in_ch2, kernel_size=1, padding=0)
            self.conv_beta2 = nn.Conv2d(1, in_ch2, kernel_size=1, padding=0)

            self.bn_concate = nn.BatchNorm2d(out_ch)

        if activ == 'relu':
            self.activation = nn.ReLU()
        elif activ == 'leaky':
            self.activation = nn.LeakyReLU(negative_slope=0.2)



    def forward(self, h1, h2, mask_h1, mask_h2):

        if hasattr(self, 'bn'):
            h1, mask_h1 = self.conv_h1(h1, mask_h1)
            h2, mask_h2 = self.conv_h2(h2, mask_h2)
            h1_ = self.pre_conv_h1(h1)
            h2_ = self.pre_conv_h2(h2)

            gamma1 = self.conv_gamma1(h1_)
            beta1 = self.conv_beta1(h1_)

            gamma2 = self.conv_gamma2(h2_)
            beta2 = self.conv_beta2(h2_)

            h1 = h1 * (1 + gamma2) + beta2
            h2 = h2 * (1 + gamma1) + beta1


        h = torch.cat([h1, h2], dim=1)
        h_mask = torch.cat([mask_h1, mask_h2], dim=1)

        h, h_mask = self.conv_concate(h, h_mask)


        if hasattr(self, 'bn'):
            h = self.bn_concate(h)
        if hasattr(self, 'activation'):
            h = self.activation(h)

        return h, h_mask


class UNet(nn.Module):
    def __init__(self, layer_size=7, input_channels=3, upsampling_mode='nearest'):
        super().__init__()
        self.freeze_enc_bn = False
        self.upsampling_mode = upsampling_mode
        self.layer_size = layer_size
        self.enc_1 = Convblock(input_channels, 64, bn=False, sample='down-7')
        self.enc_2 = Convblock(64, 128, sample='down-5')
        self.enc_3 = Convblock(128, 256, sample='down-5')
        self.enc_4 = Convblock(256, 512, sample='down-3')
        for i in range(4, self.layer_size):
            name = 'enc_{:d}'.format(i + 1)
            setattr(self, name, NONLocalBlock2D(512, 1, 512, sample='down-3'))

        for i in range(4, self.layer_size):
            name = 'dec_{:d}'.format(i + 1)
            setattr(self, name, MFT(in_ch1=512, in_ch2=512, out_ch=512, activ='leaky'))
        self.dec_4 = MFT(in_ch1=512, in_ch2=256, out_ch=256, activ='leaky')
        self.dec_3 = MFT(in_ch1=256, in_ch2=128, out_ch=128, activ='leaky')
        self.dec_2 = MFT(in_ch1=128, in_ch2=64, out_ch=64, activ='leaky')
        self.dec_1 = MFT(in_ch1=64, in_ch2=3, out_ch=input_channels,
                              bn=False, activ=None, conv_bias=True)

    def forward(self, input, input_mask):
        h_dict = {}  # for the output of enc_N
        h_mask_dict = {}  # for the output of enc_N

        h_dict['h_0'], h_mask_dict['h_0'] = input, input_mask

        h_key_prev = 'h_0'
        for i in range(1, self.layer_size + 1):
            l_key = 'enc_{:d}'.format(i)
            h_key = 'h_{:d}'.format(i)
            h_dict[h_key], h_mask_dict[h_key] = getattr(self, l_key)(
                h_dict[h_key_prev], h_mask_dict[h_key_prev])
            h_key_prev = h_key

        h_key = 'h_{:d}'.format(self.layer_size)
        h, h_mask = h_dict[h_key], h_mask_dict[h_key]

        for i in range(self.layer_size, 0, -1):
            enc_h_key = 'h_{:d}'.format(i - 1)
            dec_l_key = 'dec_{:d}'.format(i)

            h = F.interpolate(h, scale_factor=2, mode=self.upsampling_mode)
            h_mask = F.interpolate(
                h_mask, scale_factor=2, mode='nearest')

            h, h_mask = getattr(self, dec_l_key)(h1=h, h2=h_dict[enc_h_key], mask_h1=h_mask,
                                                 mask_h2=h_mask_dict[enc_h_key])

        return h, h_mask

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super().train(mode)
        if self.freeze_enc_bn:
            for name, module in self.named_modules():
                if isinstance(module, nn.BatchNorm2d) and 'enc' in name:
                    module.eval()

