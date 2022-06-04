# -*- coding:utf-8 -*-

import os

import sys
import time
import numpy as np
from tqdm import tqdm, trange
import scipy.io as scio
import random
import torch
from torch.optim import Adam
from torch.autograd import Variable
import utils
from net import NestFuse_autoencoder
from args_fusion import args
import pytorch_msssim


def main():
	original_imgs_path = utils.list_images(args.dataset)    #读取训练用图像，返回的是一个图像的集合
	train_num = 10000        #
	original_imgs_path = original_imgs_path[:train_num]     #训练中用多少图像
	# random.shuffle(original_imgs_path)            #图像随机排序
	for i in range(2,3):
		# i = 3  2到3循环，不包括3 下面作为结构相似性损失权重的下标使用
		train(i, original_imgs_path)


def train(i, original_imgs_path):

	batch_size = args.batch_size

	# load network model
	# nest_model = FusionNet_gra()
	input_nc = 1
	output_nc = 1
	deepsupervision = False  # true for deeply supervision
	nb_filter = [64, 112, 160, 208, 256]            #每一层的输出特征图通道数
	# nb_filter = [64, 64, 64, 64, 64]
	# nb_filter = [32, 64, 128, 256, 201314]

	nest_model = NestFuse_autoencoder(nb_filter, input_nc, output_nc, deepsupervision)   #nest_model现在是一个model

	if args.resume is not None:           #决定是加载模型还是训练模型
		print('Resuming, initializing using weight from {}.'.format(args.resume))     #format，用后面括号里的替换前面的
		nest_model.load_state_dict(torch.load(args.resume))       #加载新的model
	print(nest_model)        #输出新的model
	optimizer = Adam(nest_model.parameters(), args.lr)      #from torch.optim import Adam，优化器对象
	mse_loss = torch.nn.MSELoss()            #torch.nn.MSELoss()均方损失函数，loss(xi,yi)=(xi−yi)^2
	ssim_loss = pytorch_msssim.msssim         #结构相似性损失

	if args.cuda:          #初始值为1，决定是否使用GPU
		nest_model.cuda()

	tbar = trange(args.epochs)  #时期数
	print('Start training.....')         #显示任务进度条，开始训练

	Loss_pixel = []      #像素损失
	Loss_ssim = []       #结构相似性损失
	Loss_all = []        #总损失
	count_loss = 0
	all_ssim_loss = 0.
	all_pixel_loss = 0.
	for e in tbar:           #时期循环
		print('Epoch %d.....' % e)       #显示任务进度条
		# load training database        #加载训练集
		image_set_ir, batches = utils.load_dataset(original_imgs_path, batch_size)   #图像路径，batch数量
		nest_model.train()      #model训练
		count = 0
		for batch in range(batches):      #batches为batch数量
			print(e, batch)

			image_paths = image_set_ir[batch * batch_size:(batch * batch_size + batch_size)]
			img = utils.get_train_images_auto(image_paths, height=args.HEIGHT, width=args.WIDTH, flag=False)
			count += 1
			optimizer.zero_grad()       #梯度置0
			img = Variable(img, requires_grad=False)
			if args.cuda:
				img = img.cuda()
			# get fusion image
			# encoder
			en = nest_model.encoder(img)
			# decoder
			outputs = nest_model.decoder_train(en)
			# resolution loss: between fusion image and visible image
			x = Variable(img.data.clone(), requires_grad=False)

			ssim_loss_value = 0.
			pixel_loss_value = 0.
			for output in outputs:           #计算像素损失和结构相似性损失
				pixel_loss_temp = mse_loss(output, x)
				ssim_loss_temp = ssim_loss(output, x, normalize=True)
				ssim_loss_value += (1-ssim_loss_temp)
				pixel_loss_value += pixel_loss_temp
			ssim_loss_value /= len(outputs)
			pixel_loss_value /= len(outputs)

			# total loss
			total_loss = pixel_loss_value + args.ssim_weight[i] * ssim_loss_value        #
			total_loss.backward()
			optimizer.step()      #更新模型

			all_ssim_loss += ssim_loss_value.item()        #item()，得到张量中的元素
			all_pixel_loss += pixel_loss_value.item()
			if (batch + 1) % args.log_interval == 0:            #batch是第几个batch，恰好是设置的args.log_interval的倍数.设置为10
				mesg = "{}\t SSIM weight {}\tEpoch {}:\t[{}/{}]\t pixel loss: {:.6f}\t ssim loss: {:.6f}\t total: {:.6f}".format(  #\t,制表符，上下对齐。%.6f 输出小数,即保留小数点后6位
					time.ctime(), i, e + 1, count, batches,      #time.ctime()，当前时间
								  all_pixel_loss / args.log_interval,
								  (args.ssim_weight[i] * all_ssim_loss) / args.log_interval,
								  (args.ssim_weight[i] * all_ssim_loss + all_pixel_loss) / args.log_interval
				)#用像素损失，结构相似性损失，总损失分别除以args.log_interval
				tbar.set_description(mesg)       #设置进度条名称
				Loss_pixel.append(all_pixel_loss / args.log_interval)
				Loss_ssim.append(all_ssim_loss / args.log_interval)
				Loss_all.append((args.ssim_weight[i] * all_ssim_loss + all_pixel_loss) / args.log_interval)
				count_loss = count_loss + 1
				all_ssim_loss = 0.
				all_pixel_loss = 0.

			if (batch + 1) % (200 * args.log_interval) == 0:
				# save model             训练一定数量batch后开始保存模型
				nest_model.eval()
				nest_model.cpu()
				save_model_filename = args.ssim_path[i] + '/' + "Epoch_" + str(e) + "_iters_" + str(count) + "_" + \
									  str(time.ctime()).replace(' ', '_').replace(':', '_') + "_" + args.ssim_path[
										  i] + ".model"
				save_model_path = os.path.join(args.save_model_dir_autoencoder, save_model_filename)      #拼接路径
				torch.save(nest_model.state_dict(), save_model_path)       #保存模型参数字典
				# save loss data
				# pixel loss         i是结构相似性损失权重
				loss_data_pixel = Loss_pixel
				loss_filename_path = args.save_loss_dir + args.ssim_path[i] + '/' + "loss_pixel_epoch_" + str(
					args.epochs) + "_iters_" + str(count) + "_" + str(time.ctime()).replace(' ', '_').replace(':', '_') + "_" + \
									 args.ssim_path[i] + ".mat"
				scio.savemat(loss_filename_path, {'loss_pixel': loss_data_pixel})      #保存文件，loss_pixel是文件中的矩阵名，loss_data_pixel是数据
				# SSIM loss
				loss_data_ssim = Loss_ssim
				loss_filename_path = args.save_loss_dir + args.ssim_path[i] + '/' + "loss_ssim_epoch_" + str(
					args.epochs) + "_iters_" + str(count) + "_" + str(time.ctime()).replace(' ', '_').replace(':', '_') + "_" + \
									 args.ssim_path[i] + ".mat"
				scio.savemat(loss_filename_path, {'loss_ssim': loss_data_ssim})
				# all loss
				loss_data = Loss_all
				loss_filename_path = args.save_loss_dir + args.ssim_path[i] + '/' + "loss_all_epoch_" + str(e) + "_iters_" + \
									 str(count) + "-" + str(time.ctime()).replace(' ', '_').replace(':', '_') + "_" + \
									 args.ssim_path[i] + ".mat"
				scio.savemat(loss_filename_path, {'loss_all': loss_data})

				nest_model.train()
				nest_model.cuda()
				# tbar.set_description("\nCheckpoint, trained model saved at", save_model_path)

	#训练结束
	# pixel loss
	loss_data_pixel = Loss_pixel
	loss_filename_path = args.save_loss_dir + args.ssim_path[i] + '/' + "Final_loss_pixel_epoch_" + str(
		args.epochs) + "_" + str(
		time.ctime()).replace(' ', '_').replace(':', '_') + "_" + args.ssim_path[i] + ".mat"
	scio.savemat(loss_filename_path, {'final_loss_pixel': loss_data_pixel})
	loss_data_ssim = Loss_ssim
	loss_filename_path = args.save_loss_dir + args.ssim_path[i] + '/' + "Final_loss_ssim_epoch_" + str(
		args.epochs) + "_" + str(
		time.ctime()).replace(' ', '_').replace(':', '_') + "_" + args.ssim_path[i] + ".mat"
	scio.savemat(loss_filename_path, {'final_loss_ssim': loss_data_ssim})
	# SSIM loss
	loss_data = Loss_all
	loss_filename_path = args.save_loss_dir + args.ssim_path[i] + '/' + "Final_loss_all_epoch_" + str(
		args.epochs) + "_" + str(
		time.ctime()).replace(' ', '_').replace(':', '_') + "_" + args.ssim_path[i] + ".mat"
	scio.savemat(loss_filename_path, {'final_loss_all': loss_data})
	# save model
	nest_model.eval()
	nest_model.cpu()
	save_model_filename = args.ssim_path[i] + '/' "Final_epoch_" + str(args.epochs) + "_" + \
						  str(time.ctime()).replace(' ', '_').replace(':', '_') + "_" + args.ssim_path[i] + ".model"
	save_model_path = os.path.join(args.save_model_dir_autoencoder, save_model_filename)
	torch.save(nest_model.state_dict(), save_model_path)

	print("\nDone, trained model saved at", save_model_path)


def check_paths(args):
	try:
		if not os.path.exists(args.vgg_model_dir):
			os.makedirs(args.vgg_model_dir)           #创建文件夹
		if not os.path.exists(args.save_model_dir):
			os.makedirs(args.save_model_dir)
	except OSError as e:
		print(e)
		sys.exit(1)

def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False    #spend some time to find the most properly inplementation of conv for the network and be faster
    torch.backends.cudnn.deterministic = True #every time the conv algorithm returned is settled


if __name__ == "__main__":
	seed_torch(args.seed)
	main()
