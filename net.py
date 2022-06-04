import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
import OctaveConv
import pdb
import fusion_strategy

#上采样操作
class UpsampleReshape_eval(torch.nn.Module):   #继承torch.nn.Module
    def __init__(self):
        super(UpsampleReshape_eval, self).__init__() #初始化
        # self.up = nn.Upsample(scale_factor=2) #增加了一个up操作，2倍上采样
        self.up = F.interpolate

    def forward(self, x1, x2):
        x2 = self.up(x2,scale_factor=2)       #对x2进行上采样
        shape_x1 = x1.size()
        shape_x2 = x2.size()
        left = 0
        right = 0
        top = 0
        bot = 0
        #对上采样后特征图的大小进行填补，保证它和别的特征图大小相同
        if shape_x1[3] != shape_x2[3]:        #N,C,H,W
            lef_right = shape_x1[3] - shape_x2[3]          #宽之差
            if lef_right%2 is 0.0:        #宽之差是偶数
                left = int(lef_right/2)
                right = int(lef_right/2)
            else:
                left = int(lef_right / 2)
                right = int(lef_right - left)

        if shape_x1[2] != shape_x2[2]:   #高之差
            top_bot = shape_x1[2] - shape_x2[2]
            if top_bot%2 is 0.0:
                top = int(top_bot/2)
                bot = int(top_bot/2)
            else:
                top = int(top_bot / 2)
                bot = int(top_bot - top)

        reflection_padding = [left, right, top, bot]     #填充操作
        reflection_pad = nn.ReflectionPad2d(reflection_padding)
        x2 = reflection_pad(x2)
        return x2


# Convolution operation
class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, is_last=False):  #输入通道数，输出通道数，卷积核大小，步长，是否是最后一层
        super(ConvLayer, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))    #卷积前后特征图大小不变
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.dropout = nn.Dropout2d(p=0.5) #dropout防止过拟合
        self.is_last = is_last

    def forward(self, x):
        out = self.reflection_pad(x)   #填边
        out = self.conv2d(out)          #卷积
        if self.is_last is False:
            out = F.relu(out, inplace=True)  #不是最后一层就加个relu作为激活函数，inplace代表直接赋值
        return out


# light version
class DenseBlock_light(torch.nn.Module):        #两个卷积层，先将通道缩小一半，再按给的输出通道数输出
    def __init__(self, in_channels, out_channels, kernel_size, α, stride):
        super(DenseBlock_light, self).__init__()
        out_channels_def = int(in_channels / 2)     #输出通道数为输入通道数的一半
        # out_channels_def = out_channels
        denseblock = []

        denseblock += [OctaveConv.OctaveCR(in_channels, out_channels_def, kernel_size, α, stride),
                       OctaveConv.OctaveCR(out_channels_def, out_channels, (1,1), α, stride)]
        self.denseblock = nn.Sequential(*denseblock)          #？

    def forward(self, x):
        out = self.denseblock(x)
        return out


# NestFuse network - light, no desnse
class NestFuse_autoencoder(nn.Module):          #自编码器
    def __init__(self, nb_filter, input_nc=1, output_nc=1, deepsupervision=True):
        super(NestFuse_autoencoder, self).__init__()
        self.deepsupervision = deepsupervision
        block = DenseBlock_light
        output_filter = 16
        kernel_size = (3,3)
        stride = 1
        # α = 0.0625
        α = 0.125
        # α = 0.25

        self.pool = nn.MaxPool2d(2, 2)         #最大池化
        # self.up = nn.Upsample(scale_factor=2)       #上采样
        self.up = F.interpolate
        self.up_eval = UpsampleReshape_eval()        #上采样评估

        # encoder
        self.conv0 = OctaveConv.FirstOctaveCR(input_nc, output_filter, (1,1), α, stride)           #卷积操作。输入通道数。输出通道数，卷积核大小，步长
        self.DB1_0 = block(output_filter, nb_filter[0], kernel_size, α, 1)        #实例化block
        self.DB2_0 = block(nb_filter[0], nb_filter[1], kernel_size, α, 1)
        self.DB3_0 = block(nb_filter[1], nb_filter[2], kernel_size, α, 1)
        self.DB4_0 = block(nb_filter[2], nb_filter[3], kernel_size, α, 1)
        # self.DB5_0 = block(nb_filter[3], nb_filter[4], kernel_size, 1)

        # decoder
        self.DB1_1 = block(nb_filter[0] + nb_filter[1], nb_filter[0], kernel_size, α, 1)
        self.DB2_1 = block(nb_filter[1] + nb_filter[2], nb_filter[1], kernel_size, α, 1)
        self.DB3_1 = block(nb_filter[2] + nb_filter[3], nb_filter[2], kernel_size, α, 1)

        self.DB1_2 = block(nb_filter[0] * 2 + nb_filter[1], nb_filter[0], kernel_size, α, 1)
        self.DB2_2 = block(nb_filter[1] * 2 + nb_filter[2], nb_filter[1], kernel_size, α, 1)

        self.DB1_3 = block(nb_filter[0] * 3 + nb_filter[1], nb_filter[0], kernel_size, α, 1)

        if self.deepsupervision:
            self.conv1 = ConvLayer(nb_filter[0], output_nc, 1, stride)
            self.conv2 = ConvLayer(nb_filter[0], output_nc, 1, stride)
            self.conv3 = ConvLayer(nb_filter[0], output_nc, 1, stride)
        else:
            self.conv_out = OctaveConv.LastOctaveConv(nb_filter[0], output_nc, (1,1), α, stride)

        self.convh423 = ConvLayer(int(nb_filter[3] * 0.75), int(nb_filter[2] * 0.125), 1, stride)
        self.convm423 = ConvLayer(int(nb_filter[3] * 0.125), int(nb_filter[2] * 0.125), 1, stride)

        self.convh322 = ConvLayer(int(nb_filter[2] * 0.75), int(nb_filter[1] * 0.125), 1, stride)
        self.convm322 = ConvLayer(int(nb_filter[2] * 0.125), int(nb_filter[1] * 0.125), 1, stride)

        self.convh221 = ConvLayer(int(nb_filter[1] * 0.75), int(nb_filter[0] * 0.125), 1, stride)
        self.convm221 = ConvLayer(int(nb_filter[1] * 0.125), int(nb_filter[0] * 0.125), 1, stride)


    def _upsample_add(self, x, y):
        _,_,H,W = y.size()
        return F.interpolate(x, size=(H,W), mode='bilinear',align_corners=False) + y

    def encoder(self, input):
        x = self.conv0(input)
        x1_0 = self.DB1_0(x)
        x2_0 = self.DB2_0((self.pool(x1_0[0]),self.pool(x1_0[1]),self.pool(x1_0[2])))
        x3_0 = self.DB3_0((self.pool(x2_0[0]),self.pool(x2_0[1]),self.pool(x2_0[2])))
        x4_0 = self.DB4_0((self.pool(x3_0[0]),self.pool(x3_0[1]),self.pool(x3_0[2])))

        x4 = x4_0
        x3 = x3_0[0], \
             self._upsample_add(self.convh423(x4[0]), x3_0[1]), \
             self._upsample_add(self.convm423(x4[1]), x3_0[2])
        x2 = x2_0[0], \
             self._upsample_add(self.convh322(x3[0]), x2_0[1]), \
             self._upsample_add(self.convm322(x3[1]), x2_0[2])
        x1 = x1_0[0], \
             self._upsample_add(self.convh221(x2[0]), x1_0[1]), \
             self._upsample_add(self.convm221(x2[1]), x1_0[2])

        # return [x1_0, x2_0, x3_0, x4_0]
        return [x1, x2, x3, x4]
    def fusion(self, en1, en2, p_type):
        # attention weight
        fusion_function = fusion_strategy.attention_fusion_weight

        f1_0 = fusion_function(en1[0], en2[0], p_type)
        f2_0 = fusion_function(en1[1], en2[1], p_type)
        f3_0 = fusion_function(en1[2], en2[2], p_type)
        f4_0 = fusion_function(en1[3], en2[3], p_type)
        return [f1_0, f2_0, f3_0, f4_0]

    def decoder_train(self, f_en):
        x1_1 = self.DB1_1((torch.cat([f_en[0][0], self.up(f_en[1][0],scale_factor=2)], 1),torch.cat([f_en[0][1], self.up(f_en[1][1],scale_factor=2)], 1),torch.cat([f_en[0][2], self.up(f_en[1][2],scale_factor=2)], 1)))    #在1维即通道维度进行级联，传的参数不太对？

        x2_1 = self.DB2_1((torch.cat([f_en[1][0], self.up(f_en[2][0],scale_factor=2)], 1),torch.cat([f_en[1][1], self.up(f_en[2][1],scale_factor=2)], 1),torch.cat([f_en[1][2], self.up(f_en[2][2],scale_factor=2)], 1)))
        x1_2 = self.DB1_2((torch.cat([f_en[0][0], x1_1[0], self.up(x2_1[0],scale_factor=2)], 1),torch.cat([f_en[0][1], x1_1[1], self.up(x2_1[1],scale_factor=2)], 1),torch.cat([f_en[0][2], x1_1[2], self.up(x2_1[2],scale_factor=2)], 1)))

        x3_1 = self.DB3_1((torch.cat([f_en[2][0], self.up(f_en[3][0], scale_factor=2)], 1),
                           torch.cat([f_en[2][1], self.up(f_en[3][1], scale_factor=2)], 1),
                           torch.cat([f_en[2][2], self.up(f_en[3][2], scale_factor=2)], 1)))
        x2_2 = self.DB2_2((torch.cat([f_en[1][0], x2_1[0], self.up(x3_1[0], scale_factor=2)], 1),
                           torch.cat([f_en[1][1], x2_1[1], self.up(x3_1[1], scale_factor=2)], 1),
                           torch.cat([f_en[1][2], x2_1[2], self.up(x3_1[2], scale_factor=2)], 1)))
        x1_3 = self.DB1_3((torch.cat([f_en[0][0], x1_1[0], x1_2[0], self.up(x2_2[0], scale_factor=2)], 1),
                           torch.cat([f_en[0][1], x1_1[1], x1_2[1], self.up(x2_2[1], scale_factor=2)], 1),
                           torch.cat([f_en[0][2], x1_1[2], x1_2[2], self.up(x2_2[2], scale_factor=2)], 1)
                           ))

        # x3_1 = self.DB3_1((torch.cat([f_en[2][0], self.up(f_en[3][0],scale_factor=2)], 1),torch.cat([f_en[2][1], self.up(f_en[3][1],scale_factor=2)], 1)))
        # x2_2 = self.DB2_2((torch.cat([f_en[1][0], self.up(x3_1[0],scale_factor=2)], 1),torch.cat([f_en[1][1], self.up(x3_1[1],scale_factor=2)], 1)))
        #
        # x1_3 = self.DB1_3((torch.cat([f_en[0][0], self.up(x2_2[0],scale_factor=2)], 1),torch.cat([f_en[0][1], self.up(x2_2[1],scale_factor=2)], 1)))

        if self.deepsupervision:          #没看懂
            output1 = self.conv1(x1_1)
            output2 = self.conv2(x1_2)
            output3 = self.conv3(x1_3)
            # output4 = self.conv4(x1_4)
            return [output1, output2, output3]
        else:
            output = self.conv_out(x1_3)
            return [output]

    def decoder_eval(self, f_en):

        # x1_1 = self.DB1_1(torch.cat([f_en[0], self.up_eval(f_en[0], f_en[1])], 1))
        #
        # x2_1 = self.DB2_1(torch.cat([f_en[1], self.up_eval(f_en[1], f_en[2])], 1))
        # x1_2 = self.DB1_2(torch.cat([f_en[0], x1_1, self.up_eval(f_en[0], x2_1)], 1))
        #
        # x3_1 = self.DB3_1(torch.cat([f_en[2], self.up_eval(f_en[2], f_en[3])], 1))
        # x2_2 = self.DB2_2(torch.cat([f_en[1], x2_1, self.up_eval(f_en[1], x3_1)], 1))
        # x1_3 = self.DB1_3(torch.cat([f_en[0], x1_1, x1_2, self.up_eval(f_en[0], x2_2)], 1))

        x1_1 = self.DB1_1((torch.cat([f_en[0][0], self.up_eval(f_en[0][0], f_en[1][0])], 1),torch.cat([f_en[0][1], self.up_eval(f_en[0][1], f_en[1][1])], 1),torch.cat([f_en[0][2], self.up_eval(f_en[0][2], f_en[1][2])], 1)))  # 在1维即通道维度进行级联，传的参数不太对？
        x2_1 = self.DB2_1((torch.cat([f_en[1][0], self.up_eval(f_en[1][0], f_en[2][0])], 1),torch.cat([f_en[1][1], self.up_eval(f_en[1][1], f_en[2][1])], 1),torch.cat([f_en[1][2], self.up_eval(f_en[1][2], f_en[2][2])], 1)))
        x1_2 = self.DB1_2((torch.cat([f_en[0][0], x1_1[0], self.up_eval(f_en[0][0], x2_1[0])], 1),torch.cat([f_en[0][1], x1_1[1], self.up_eval(f_en[0][1], x2_1[1])], 1),torch.cat([f_en[0][2], x1_1[2], self.up_eval(f_en[0][2], x2_1[2])], 1)))
        x3_1 = self.DB3_1((torch.cat([f_en[2][0], self.up_eval(f_en[2][0], f_en[3][0])], 1),
                           torch.cat([f_en[2][1], self.up_eval(f_en[2][1], f_en[3][1])], 1),
                           torch.cat([f_en[2][2], self.up_eval(f_en[2][2], f_en[3][2])], 1)))
        x2_2 = self.DB2_2((torch.cat([f_en[1][0], x2_1[0], self.up_eval(f_en[1][0], x3_1[0])], 1),
                           torch.cat([f_en[1][1], x2_1[1], self.up_eval(f_en[1][1], x3_1[1])], 1),
                           torch.cat([f_en[1][2], x2_1[2], self.up_eval(f_en[1][2], x3_1[2])], 1)))
        x1_3 = self.DB1_3((torch.cat([f_en[0][0], x1_1[0], x1_2[0], self.up_eval(f_en[0][0], x2_2[0])], 1),
                           torch.cat([f_en[0][1], x1_1[1], x1_2[1], self.up_eval(f_en[0][1], x2_2[1])], 1),
                           torch.cat([f_en[0][2], x1_1[2], x1_2[2], self.up_eval(f_en[0][2], x2_2[2])], 1)))
        # x3_1 = self.DB3_1((torch.cat([f_en[2][0], self.up_eval(f_en[2][0], f_en[3][0])], 1),torch.cat([f_en[2][1], self.up_eval(f_en[2][1], f_en[3][1])], 1)))
        # x2_2 = self.DB2_2((torch.cat([f_en[1][0], self.up_eval(f_en[1][0], x3_1[0])], 1),torch.cat([f_en[1][1], self.up_eval(f_en[1][1], x3_1[1])], 1)))
        # x1_3 = self.DB1_3((torch.cat([f_en[0][0], self.up_eval(f_en[0][0], x2_2[0])], 1),torch.cat([f_en[0][1], self.up_eval(f_en[0][1], x2_2[1])], 1)))

        if self.deepsupervision:
            output1 = self.conv1(x1_1)
            output2 = self.conv2(x1_2)
            output3 = self.conv3(x1_3)
            return [output1, output2, output3]
        else:
            output = self.conv_out(x1_3)
            return [output]