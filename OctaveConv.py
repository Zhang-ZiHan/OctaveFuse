#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Xiangtai Li(lxtpku@pku.edu.cn)
# Pytorch Implementation of Octave Conv Operation
# This version use nn.Conv2d because alpha_in always equals alpha_out

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import net

class OctaveConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, alpha=0.5, stride=1, padding=0, dilation=1,
                 groups=1, bias=False):
        super(OctaveConv, self).__init__()
        kernel_size = kernel_size[0]
        self.pool = nn.AdaptiveAvgPool2d

        reflection_padding = int(np.floor(kernel_size / 2))  # 卷积前后特征图大小不变
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)

        self.upsample = F.interpolate
        self.stride = stride

        self.a2a = torch.nn.Conv2d(in_channels - 2 * int(alpha * in_channels),
                                   out_channels - 2 * int(alpha * out_channels),
                                   kernel_size, 1, padding, dilation, groups, bias)
        self.b2b = torch.nn.Conv2d(int(alpha * in_channels), int(alpha * out_channels),
                                   kernel_size, 1, padding, dilation, groups, bias)
        self.c2c = torch.nn.Conv2d(int(alpha * in_channels), int(alpha * out_channels),
                                   kernel_size, 1, padding, dilation, groups, bias)
        self.a2b = torch.nn.Conv2d(in_channels - 2 * int(alpha * in_channels), int(alpha * out_channels),
                                   kernel_size, 1, padding, dilation, groups, bias)
        self.b2a = torch.nn.Conv2d(int(alpha * in_channels), out_channels - 2 * int(alpha * out_channels),
                                   kernel_size, 1, padding, dilation, groups, bias)
        self.b2c = torch.nn.Conv2d(int(alpha * in_channels), int(alpha * out_channels),
                                   kernel_size, 1, padding, dilation, groups, bias)
        self.c2b = torch.nn.Conv2d(int(alpha * in_channels), int(alpha * out_channels),
                                   kernel_size, 1, padding, dilation, groups, bias)

    def forward(self, x):
        a, b, c = x
        _, _, H_a, W_a = a.size()
        _, _, H_b, W_b = b.size()
        _, _, H_c, W_c = c.size()
        self.a2b_pool = self.pool((int(H_b), int(W_b)))
        self.b2c_pool = self.pool((int(H_c), int(W_c)))

        a2b = self.a2b_pool(a)
        b2c = self.b2c_pool(b)

        a = self.reflection_pad(a)
        b = self.reflection_pad(b)
        c = self.reflection_pad(c)
        a2b = self.reflection_pad(a2b)
        b2c = self.reflection_pad(b2c)

        if self.stride ==2:
            X_h, X_l = self.h2g_pool(X_h), self.h2g_pool(X_l)

        a2a = self.a2a(a)
        b2b = self.b2b(b)
        c2c = self.c2c(c)
        a2b = self.a2b(a2b)
        b2c = self.b2c(b2c)
        b2a = self.upsample(self.b2a(b), size=(int(H_a), int(W_a)), mode='bilinear', align_corners=False)
        c2b = self.upsample(self.c2b(c), size=(int(H_b), int(W_b)), mode='bilinear', align_corners=False)

        a = a2a + b2a
        b = b2b + a2b +c2b
        c = c2c + b2c

        return a, b, c


class FirstOctaveConv(nn.Module):
    def __init__(self, in_channels, out_channels,kernel_size, alpha=0.5, stride=1, padding=0, dilation=1,
                 groups=1, bias=False):
        super(FirstOctaveConv, self).__init__()
        self.stride = stride
        kernel_size = kernel_size[0]
        self.pool = nn.AdaptiveAvgPool2d
        self.a2a = torch.nn.Conv2d(in_channels, out_channels - 2 * int(alpha * out_channels),
                                   kernel_size, 1, padding, dilation, groups, bias)
        self.b2b = torch.nn.Conv2d(in_channels, int(alpha * out_channels),
                                   kernel_size, 1, padding, dilation, groups, bias)
        self.c2c = torch.nn.Conv2d(in_channels, int(alpha * out_channels),
                                   kernel_size, 1, padding, dilation, groups, bias)

    def forward(self, x):
        if self.stride ==2:
            x = self.h2g_pool(x)
        _, _, H, W = x.size()
        # a,b,c分别是卷积的三层
        self.a2b_pool = self.pool((int(0.75 * H), int(0.75 * W)))
        self.a2c_pool = self.pool((int(0.5 * H), int(0.5 * W)))

        a = self.a2a(x)
        b = self.b2b(self.a2b_pool(x))
        c = self.c2c(self.a2c_pool(x))

        return a, b, c


class LastOctaveConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, alpha=0.5, stride=1, padding=0, dilation=1,
                 groups=1, bias=False):
        super(LastOctaveConv, self).__init__()
        self.stride = stride
        kernel_size = kernel_size[0]
        self.upsample = F.interpolate

        self.b2a = torch.nn.Conv2d(int(alpha * in_channels), out_channels,
                                   kernel_size, 1, padding, dilation, groups, bias)
        self.c2a = torch.nn.Conv2d(int(alpha * in_channels), out_channels,
                                   kernel_size, 1, padding, dilation, groups, bias)
        self.a2a = torch.nn.Conv2d(in_channels - 2 * int(alpha * in_channels),
                                   out_channels,
                                   kernel_size, 1, padding, dilation, groups, bias)

    def forward(self, x):
        a, b, c = x
        _, _, H, W = a.size()

        if self.stride ==2:
            X_h, X_l = self.h2g_pool(X_h), self.h2g_pool(X_l)

        a2a = self.a2a(a)
        b2a = self.upsample(self.b2a(b), size=(int(H), int(W)), mode='bilinear', align_corners=False)
        c2a = self.upsample(self.c2a(c), size=(int(H), int(W)), mode='bilinear', align_corners=False)
        
        a = a2a + b2a +c2a

        return a


class OctaveCBR(nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size=(3,3),alpha=0.5, stride=1, padding=1, dilation=1,
                 groups=1, bias=False, norm_layer=nn.BatchNorm2d):
        super(OctaveCBR, self).__init__()
        self.conv = OctaveConv(in_channels,out_channels,kernel_size, alpha, stride, padding, dilation, groups, bias)
        self.bn_h = norm_layer(int(out_channels*(1-alpha)))
        self.bn_l = norm_layer(int(out_channels*alpha))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x_h, x_l = self.conv(x)
        x_h = self.relu(self.bn_h(x_h))
        x_l = self.relu(self.bn_l(x_l))
        return x_h, x_l


class OctaveCB(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3,3), alpha=0.5, stride=1, padding=1, dilation=1,
                 groups=1, bias=False, norm_layer=nn.BatchNorm2d):
        super(OctaveCB, self).__init__()
        self.conv = OctaveConv(in_channels, out_channels, kernel_size, alpha, stride, padding, dilation,
                               groups, bias)
        self.bn_h = norm_layer(int(out_channels * (1 - alpha)))
        self.bn_l = norm_layer(int(out_channels * alpha))

    def forward(self, x):
        x_h, x_l = self.conv(x)
        x_h = self.bn_h(x_h)
        x_l = self.bn_l(x_l)
        return x_h, x_l


class FirstOctaveCBR(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3,3),alpha=0.5, stride=1, padding=1, dilation=1,
                 groups=1, bias=False,norm_layer=nn.BatchNorm2d):
        super(FirstOctaveCBR, self).__init__()
        self.conv = FirstOctaveConv(in_channels,out_channels,kernel_size, alpha,stride,padding,dilation,groups,bias)
        self.bn_h = norm_layer(int(out_channels * (1 - alpha)))
        self.bn_l = norm_layer(int(out_channels * alpha))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x_h, x_l = self.conv(x)
        x_h = self.relu(self.bn_h(x_h))
        x_l = self.relu(self.bn_l(x_l))
        return x_h, x_l


class LastOCtaveCBR(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3,3), alpha=0.5, stride=1, padding=1, dilation=1,
                 groups=1, bias=False, norm_layer=nn.BatchNorm2d):
        super(LastOCtaveCBR, self).__init__()
        self.conv = LastOctaveConv(in_channels, out_channels, kernel_size, alpha, stride, padding, dilation, groups, bias)
        self.bn_h = norm_layer(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x_h = self.conv(x)
        x_h = self.relu(self.bn_h(x_h))
        return x_h


class FirstOctaveCB(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3,3), alpha=0.5,stride=1, padding=1, dilation=1,
                 groups=1, bias=False, norm_layer=nn.BatchNorm2d):
        super(FirstOctaveCB, self).__init__()
        self.conv = FirstOctaveConv(in_channels,out_channels,kernel_size, alpha,stride,padding,dilation,groups,bias)
        self.bn_h = norm_layer(int(out_channels * (1 - alpha)))
        self.bn_l = norm_layer(int(out_channels * alpha))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x_h, x_l = self.conv(x)
        x_h = self.bn_h(x_h)
        x_l = self.bn_l(x_l)
        return x_h, x_l


class LastOCtaveCB(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, alpha=0.5, stride=1, padding=1, dilation=1,
                 groups=1, bias=False, norm_layer=nn.BatchNorm2d):
        super(LastOCtaveCB, self).__init__()
        self.conv = LastOctaveConv( in_channels, out_channels, kernel_size, alpha, stride, padding, dilation, groups, bias)
        self.bn_h = norm_layer(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x_h = self.conv(x)
        x_h = self.bn_h(x_h)
        return x_h

class FirstOctaveCR(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3,3),alpha=0.5, stride=1, padding=0, dilation=1,
                 groups=1, bias=False,norm_layer=nn.BatchNorm2d):
        super(FirstOctaveCR, self).__init__()
        self.conv = FirstOctaveConv(in_channels,out_channels,kernel_size, alpha,stride,padding,dilation,groups,bias)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        a, b, c = self.conv(x)
        a = self.relu(a)
        b = self.relu(b)
        c = self.relu(c)
        return a, b, c

class OctaveCR(nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size=(3,3),alpha=0.5, stride=1, padding=0, dilation=1,
                 groups=1, bias=False, norm_layer=nn.BatchNorm2d):
        super(OctaveCR, self).__init__()
        self.conv = OctaveConv(in_channels,out_channels,kernel_size, alpha, stride, padding, dilation, groups, bias)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        a, b, c = self.conv(x)
        a = self.relu(a)
        b = self.relu(b)
        c = self.relu(c)
        return a, b, c


if __name__ == '__main__':
    # nn.Conv2d
    high = torch.Tensor(1, 64, 32, 32).cuda()
    low = torch.Tensor(1, 192, 16, 16).cuda()
    # test Oc conv
    OCconv = OctaveConv(kernel_size=(3,3),in_channels=256,out_channels=512,bias=False,stride=2,alpha=0.75).cuda()
    i = high,low
    x_out,y_out = OCconv(i)
    print(x_out.size())
    print(y_out.size())

    i = torch.Tensor(1, 3, 512, 512).cuda()
    FOCconv = FirstOctaveConv(kernel_size=(3, 3), in_channels=3, out_channels=128).cuda()
    x_out, y_out = FOCconv(i)
    print("First: ", x_out.size(), y_out.size())
    # test last Octave Cov
    LOCconv = LastOctaveConv(kernel_size=(3, 3), in_channels=256, out_channels=128, alpha=0.75).cuda()
    i = high, low
    out = LOCconv(i)
    print("Last: ", out.size())

    # test OCB
    ocb = OctaveCB(in_channels=256, out_channels=128, alpha=0.75).cuda()
    i = high, low
    x_out_h, y_out_l = ocb(i)
    print("OCB:",x_out_h.size(),y_out_l.size())

    # test last OCB
    ocb_last = LastOCtaveCBR(256, 128, alpha=0.75).cuda()
    i = high, low
    x_out_h = ocb_last(i)
    print("Last OCB", x_out_h.size())
