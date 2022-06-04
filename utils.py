import os
import random
import numpy as np
import torch
from PIL import Image
from args_fusion import args
from scipy.misc import imread, imsave, imresize
import matplotlib as mpl

from os import listdir
from os.path import join

def list_images(directory):        #图像目录
    images = []
    names = []
    dir = listdir(directory)      #返回所给路径所包含的文件或文件夹
    dir.sort()                 #排序
    for file in dir:
        name = file.lower()      #小写
        if name.endswith('.png'):
            images.append(join(directory, file))      #路径+文件名 join用于链接字符串
        elif name.endswith('.jpg'):
            images.append(join(directory, file))
        elif name.endswith('.jpeg'):
            images.append(join(directory, file))
        name1 = name.split('.')        #如果名字里带‘.’,则以第一个点前面的名字作为全名
        names.append(name1[0])
    return images


def tensor_load_rgbimage(filename, size=None, scale=None, keep_asp=False):     #加载rgb图像转化为tensor
    img = Image.open(filename).convert('RGB')         #读出文件，转化为RGB图像
    if size is not None:
        if keep_asp:
            size2 = int(size * 1.0 / img.size[0] * img.size[1])         #size为图像的长
            img = img.resize((size, size2), Image.ANTIALIAS)          #高质量的转变图像大小
        else:
            img = img.resize((size, size), Image.ANTIALIAS)            #图像长宽一样

    elif scale is not None:
        img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)), Image.ANTIALIAS)
    img = np.array(img).transpose(2, 0, 1)          #对图像矩阵转置,通道数，高，宽
    img = torch.from_numpy(img).float()          #数组转化为张量
    return img


def tensor_save_rgbimage(tensor, filename, cuda=False):         #保存rgb图像
    if cuda:
        # img = tensor.clone().cpu().clamp(0, 255).numpy()
        img = tensor.cpu().clamp(0, 255).data[0].numpy()   #GPU的tensor转化成numpy，限制0~255，0维
    else:
        # img = tensor.clone().clamp(0, 255).numpy()
        img = tensor.clamp(0, 255).numpy()                 #tensor转化成numpy，限制0~255
    img = img.transpose(1, 2, 0).astype('uint8')          #矩阵转置
    img = Image.fromarray(img)                            #array转换成image
    img.save(filename)                                    #保存图像

#OpenCV里面的默认存储方式虽然写作RGB但是实际上存储是BGR图像
def tensor_save_bgrimage(tensor, filename, cuda=False):
    (b, g, r) = torch.chunk(tensor, 3)             #按长分开？torch.chunk把一个tensor均匀分割成若干个小tensor
    tensor = torch.cat((r, g, b))
    tensor_save_rgbimage(tensor, filename, cuda)     #变成rgb再保存


def gram_matrix(y):                #格拉姆矩阵
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)          #view()函数作用是将一个多行的Tensor,拼接成一行。每个通道特征图变成一维向量
    features_t = features.transpose(1, 2)        #转置ch和w*h
    gram = features.bmm(features_t) / (ch * h * w)         #计算两个tensor的矩阵乘法，torch.bmm(a,b),tensor a 的size为(b,h,w),tensor b的size为(b,w,h),注意两个tensor的维度必须为3.
    return gram


def matSqrt(x):
    U,D,V = torch.svd(x)          #奇异值分解
    return U * (D.pow(0.5).diag()) * V.t()


# load training images
def load_dataset(image_path, BATCH_SIZE, num_imgs=None):      #加载训练集
    if num_imgs is None:
        num_imgs = len(image_path)
    original_imgs_path = image_path[:num_imgs]
    # random
    random.shuffle(original_imgs_path)         #打乱图像顺序
    mod = num_imgs % BATCH_SIZE           #mod为batch数量
    print('BATCH SIZE %d.' % BATCH_SIZE)
    print('Train images number %d.' % num_imgs)
    print('Train images samples %s.' % str(num_imgs / BATCH_SIZE))

    if mod > 0:
        print('Train set has been trimmed %d samples...\n' % mod)
        original_imgs_path = original_imgs_path[:-mod]
    batches = int(len(original_imgs_path) // BATCH_SIZE)        #//表示商向下取整，batches现在为batch数量
    return original_imgs_path, batches


def get_image(path, height=256, width=256, flag=False):         #加载图像
    if flag is True:
        image = imread(path, mode='RGB')
    else:
        image = imread(path, mode='L')

    if height is not None and width is not None:
        image = imresize(image, [height, width], interp='nearest')
    return image


# load images - test phase
def get_test_image(paths, height=None, width=None, flag=False):       #得到测试图像
    if isinstance(paths, str):            #isinstance（）如果对象的类型与参数二的类型（classinfo）相同则返回 True，否则返回 False
        paths = [paths]
    images = []
    for path in paths:
        image = imread(path, mode='L')
        if height is not None and width is not None:
            image = imresize(image, [height, width], interp='nearest')

        base_size = 512
        h = image.shape[0]
        w = image.shape[1]
        c = 1
        if h > base_size or w > base_size:        #比较大的图像就直接分块
            c = 4
            images = get_img_parts(image, h, w)
        else:
            image = np.reshape(image, [1, image.shape[0], image.shape[1]])
            images.append(image)
            images = np.stack(images, axis=0)       #stack(x,axis=0)这个函数用于将多个数组合并，其中axis表示在第几个空间开始进行结合
            images = torch.from_numpy(images).float()         #torch.from_numpy完成数组numpy到tensor的转换

    # images = np.stack(images, axis=0)
    # images = torch.from_numpy(images).float()
    return images, h, w, c


def get_img_parts(image, h, w):        #分割图像
    images = []
    h_cen = int(np.floor(h / 2))        #floor() 返回数字的下舍整数
    w_cen = int(np.floor(w / 2))
    img1 = image[0:h_cen + 3, 0: w_cen + 3]
    img1 = np.reshape(img1, [1, 1, img1.shape[0], img1.shape[1]])
    img2 = image[0:h_cen + 3, w_cen - 2: w]
    img2 = np.reshape(img2, [1, 1, img2.shape[0], img2.shape[1]])
    img3 = image[h_cen - 2:h, 0: w_cen + 3]
    img3 = np.reshape(img3, [1, 1, img3.shape[0], img3.shape[1]])
    img4 = image[h_cen - 2:h, w_cen - 2: w]
    img4 = np.reshape(img4, [1, 1, img4.shape[0], img4.shape[1]])
    images.append(torch.from_numpy(img1).float())
    images.append(torch.from_numpy(img2).float())
    images.append(torch.from_numpy(img3).float())
    images.append(torch.from_numpy(img4).float())
    return images


def recons_fusion_images(img_lists, h, w):        #将分成四块的图像重构
    img_f_list = []
    h_cen = int(np.floor(h / 2))
    w_cen = int(np.floor(w / 2))
    ones_temp = torch.ones(1, 1, h, w).cuda()     #torch.ones返回一个全为1 的张量
    for i in range(len(img_lists[0])):
        # img1, img2, img3, img4
        img1 = img_lists[0][i]
        img2 = img_lists[1][i]
        img3 = img_lists[2][i]
        img4 = img_lists[3][i]

        # save_image_test(img1, './outputs/test/block1.png')
        # save_image_test(img2, './outputs/test/block2.png')
        # save_image_test(img3, './outputs/test/block3.png')
        # save_image_test(img4, './outputs/test/block4.png')

        img_f = torch.zeros(1, 1, h, w).cuda()        #返回一个全为标量 0 的张量
        count = torch.zeros(1, 1, h, w).cuda()

        img_f[:, :, 0:h_cen + 3, 0: w_cen + 3] += img1
        count[:, :, 0:h_cen + 3, 0: w_cen + 3] += ones_temp[:, :, 0:h_cen + 3, 0: w_cen + 3]
        img_f[:, :, 0:h_cen + 3, w_cen - 2: w] += img2
        count[:, :, 0:h_cen + 3, w_cen - 2: w] += ones_temp[:, :, 0:h_cen + 3, w_cen - 2: w]
        img_f[:, :, h_cen - 2:h, 0: w_cen + 3] += img3
        count[:, :, h_cen - 2:h, 0: w_cen + 3] += ones_temp[:, :, h_cen - 2:h, 0: w_cen + 3]
        img_f[:, :, h_cen - 2:h, w_cen - 2: w] += img4
        count[:, :, h_cen - 2:h, w_cen - 2: w] += ones_temp[:, :, h_cen - 2:h, w_cen - 2: w]
        img_f = img_f / count
        img_f_list.append(img_f)
    return img_f_list


def save_image_test(img_fusion, output_path):          #保存测试图像
    img_fusion = img_fusion.float()
    if args.cuda:
        img_fusion = img_fusion.cpu().data[0].numpy()
        # img_fusion = img_fusion.cpu().clamp(0, 255).data[0].numpy()
    else:
        img_fusion = img_fusion.clamp(0, 255).data[0].numpy()

    img_fusion = (img_fusion - np.min(img_fusion)) / (np.max(img_fusion) - np.min(img_fusion))
    img_fusion = img_fusion * 255
    img_fusion = img_fusion.transpose(1, 2, 0).astype('uint8')
    # cv2.imwrite(output_path, img_fusion)
    if img_fusion.shape[2] == 1:
        img_fusion = img_fusion.reshape([img_fusion.shape[0], img_fusion.shape[1]])
    # 	img_fusion = imresize(img_fusion, [h, w])
    imsave(output_path, img_fusion)

def save_image_test_temp(a, output_path):
    # a[i] = a[i].mean(dim=1, keepdim=True)
    a = a.mean(dim=1, keepdim=True)

    a = a.float()
    a = a.cpu().data[0].numpy()

    a = (a - np.min(a)) / (np.max(a) - np.min(a))
    a = a * 255
    # a[a < 0.5] = 0
    # a[a >= 0.5] = 255

    a = a.transpose(1, 2, 0).astype('uint8')
    # cv2.imwrite(output_path, img_fusion)
    if a.shape[2] == 1:
        a = a.reshape([a.shape[0], a.shape[1]])
    # 	img_fusion = imresize(img_fusion, [h, w])
    imsave(output_path, a)


def get_train_images(paths, height=256, width=256, flag=False):          #加载训练图像
    if isinstance(paths, str):
        paths = [paths]
    images_ir = []            #ir红外
    images_vi = []            #vi可视
    for path in paths:
        image = get_image(path, height, width, flag)
        image = np.reshape(image, [1, height, width])
        # imsave('./outputs/ir_gray.jpg', image)
        # image = image.transpose(2, 0, 1)
        images_ir.append(image)

        path_vi = path.replace('lwir', 'visible')
        image = get_image(path_vi, height, width, flag)
        image = np.reshape(image, [1, height, width])
        # imsave('./outputs/vi_gray.jpg', image)
        # image = image.transpose(2, 0, 1)
        images_vi.append(image)

    images_ir = np.stack(images_ir, axis=0)
    images_ir = torch.from_numpy(images_ir).float()

    images_vi = np.stack(images_vi, axis=0)
    images_vi = torch.from_numpy(images_vi).float()
    return images_ir, images_vi


def get_train_images_auto(paths, height=256, width=256, flag=False):       #自动加载训练图像
    if isinstance(paths, str):
        paths = [paths]
    images = []
    for path in paths:
        image = get_image(path, height, width, flag)
        if flag is True:
            image = np.transpose(image, (2, 0, 1))
        else:
            image = np.reshape(image, [1, height, width])
        images.append(image)

    images = np.stack(images, axis=0)
    images = torch.from_numpy(images).float()
    return images


# 自定义colormap
def colormap():
    return mpl.colors.LinearSegmentedColormap.from_list('cmap', ['#FFFFFF', '#98F5FF', '#00FF00', '#FFFF00','#FF0000', '#8B0000'], 256)




