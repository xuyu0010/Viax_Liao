# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 20:14:56 2020

@author: lcx98
"""

"""
Author: Yunpeng Chen
"""
import logging
import os
from collections import OrderedDict

import torch.nn as nn
import torch

try:
	from . import initializer
except:
	import initializer

try:
	from .mfnet_base import MF_UNIT
except:
	from mfnet_base import MF_UNIT

class Motion_Exctractor_Max(nn.Module):
	def __init__(self, inplanes, iterplanes, outplanes, num_embedding=20):

		super(Motion_Exctractor_Max, self).__init__()

		# self.mh_up = nn.Sequential(OrderedDict([
					# ('linear', nn.Linear(2, num_embedding))
					# ('bn', nn.BatchNorm2d(inplanes))
					# ]))
		self.mh_up = nn.Linear(2, num_embedding)

		self.mh_conv1 = nn.Sequential(OrderedDict([
					('conv', nn.Conv2d( inplanes, iterplanes, kernel_size=(1,3), padding=(0,1), stride=(1,2), bias=False)),
					('bn', nn.BatchNorm2d(iterplanes)),
					('tanh', nn.Tanh())
					# ('relu', nn.ReLU(inplace=True))
					]))

		self.mh_conv2 = nn.Sequential(OrderedDict([
					('conv', nn.Conv2d( iterplanes, outplanes, kernel_size=(1,3), padding=(0,1), stride=(1,2), bias=False)),
					('bn', nn.BatchNorm2d(outplanes)),
					('tanh', nn.Tanh())
					# ('relu', nn.ReLU(inplace=True))
					]))

		self.mh_pool = nn.AvgPool2d(kernel_size=(1,5), stride=(1,1))

	def forward(self, h):

		N = h.shape[0]
		print(N)
		C = h.shape[1]
		F = h.shape[2]
		W = h.shape[3]

		dim_total = N * C * F
		h_flatten = h.view(N, C, F, -1)
		indices_h = torch.argmax(h_flatten, dim=3).view(dim_total, -1)
		indices_h = torch.cat(((indices_h / W).view(-1,1), (indices_h % W).view(-1,1)), dim=1).view(N, C, F, -1)
		
		motion_h = (indices_h[:,:,1:F,:] - indices_h[:,:,0:(F-1),:]).float()
		motion_h = self.mh_up(motion_h)					#  7x2 -> 7x14
		motion_h = self.mh_conv1(motion_h)				# 7x14 -> 7x7 
		motion_h = self.mh_conv2(motion_h)				# 7x7  -> 7x4
		motion_h = self.mh_pool(motion_h)

		return motion_h


class Motion_Exctractor_MEAN(nn.Module):
	def __init__(self, inplanes, num_embedding=20):

		super(Motion_Exctractor_MEAN, self).__init__()

		self.mh_max_up = nn.Linear(2, num_embedding)
		self.mh_min_up = nn.Linear(2, num_embedding)

		self.mh_conv1 = nn.Sequential(OrderedDict([
					('conv', nn.Conv3d( inplanes, inplanes, kernel_size=1, stride=1, bias=False)),
					('bn', nn.BatchNorm3d(inplanes)),
					('tanh', nn.Tanh())
					# ('relu', nn.ReLU(inplace=True))
					]))

	def forward(self, h):

		N = h.shape[0]
		C = h.shape[1]
		F = h.shape[2]
		W = h.shape[3]

		dim_total = N * C * F
		h_flatten = h.view(N, C, F, -1)
		
		h_maxk = max((1,5))  
		h_top5_max, pred = h_flatten.topk(h_maxk, 3, True, True)
#		print(h_maxk)
#		print(h_top5_max.shape)
		h_top5_max = h_top5_max.float()
		h_max_mean=torch.mean(h_top5_max,dim=3,keepdim=True).view(dim_total,-1)
#		print(h_max_mean.shape)
		
		h_mink = min((1,5))  
		h_top5_min, pred = h_flatten.topk(h_mink, 3, True, True)
#		print(h_mink)
#		print(h_top5_min.shape)
		h_top5_min = h_top5_min.float()
		h_min_mean=torch.mean(h_top5_min,dim=3,keepdim=True).view(dim_total,-1)
#		print(h_min_mean.shape)
		
		
#        a = h_max_mean / W
#        b = h_max_mean % W
#        c = h_min_mean / W
#        d = h_min_mean % W
#        print(a)
#        print(a.shape)
#        print(b)
#        print(b.shape)
#        print(c)
#        print(c.shape)
#        print(d)
#        print(d.shape)
#        print((h_max_mean / W).view(-1,1).shape)
#        print(torch.cat(((h_max_mean / W).view(-1,1), (h_max_mean % W).view(-1,1)), dim=1).shape)
		indices_h_max = torch.cat(((h_max_mean / W).view(-1,1), (h_max_mean % W).view(-1,1)), dim=1).view(N, C, F, 1, -1)
		indices_h_min = torch.cat(((h_min_mean / W).view(-1,1), (h_min_mean % W).view(-1,1)), dim=1).view(N, C, F, 1, -1)
		indices_h = torch.cat((indices_h_max,indices_h_min), dim=3)
#        print(indices_h_max.shape)
#        print(indices_h.shape)
		
		motion_h = self.mh_max_up(indices_h.float()).permute(0,1,2,4,3)		# 8x2x2  ->  8x2x14
#        print(mh_max_up(indices_h.float()).shape)
#        print(motion_h.shape)
#        print(motion_h)
		motion_h = self.mh_min_up(motion_h.float())							# 8x14x2 ->  8x14x14
		motion_h = self.mh_conv1(motion_h)			
        
		return motion_h


class MFNET_SP_LINEAR_DUAL(nn.Module):

	def __init__(self, num_classes, pretrained=False, **kwargs):
		super(MFNET_SP_LINEAR_DUAL, self).__init__()

		groups = 16
		k_sec  = {  2: 3, \
					3: 4, \
					4: 6, \
					5: 3  }

		# conv1 - x224 (x16)
		conv1_num_out = 16
		self.conv1 = nn.Sequential(OrderedDict([
					('conv', nn.Conv3d( 3, conv1_num_out, kernel_size=(3,5,5), padding=(1,2,2), stride=(1,2,2), bias=False)),
					('bn', nn.BatchNorm3d(conv1_num_out)),
					('relu', nn.ReLU(inplace=True))
					]))
		self.maxpool = nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1))

		# conv2 - x56 (x8)
		num_mid = 96
		conv2_num_out = 96
		self.conv2 = nn.Sequential(OrderedDict([
					("B%02d"%i, MF_UNIT(num_in=conv1_num_out if i==1 else conv2_num_out,
										num_mid=num_mid,
										num_out=conv2_num_out,
										stride=(2,1,1) if i==1 else (1,1,1),
										g=groups,
										first_block=(i==1))) for i in range(1,k_sec[2]+1)
					]))

		# conv3 - x28 (x8)
		num_mid *= 2
		conv3_num_out = 2 * conv2_num_out
		self.conv3 = nn.Sequential(OrderedDict([
					("B%02d"%i, MF_UNIT(num_in=conv2_num_out if i==1 else conv3_num_out,
										num_mid=num_mid,
										num_out=conv3_num_out,
										stride=(1,2,2) if i==1 else (1,1,1),
										g=groups,
										first_block=(i==1))) for i in range(1,k_sec[3]+1)
					]))

		# conv4 - x14 (x8)
		num_mid *= 2
		conv4_num_out = 2 * conv3_num_out
		self.conv4 = nn.Sequential(OrderedDict([
					("B%02d"%i, MF_UNIT(num_in=conv3_num_out if i==1 else conv4_num_out,
										num_mid=num_mid,
										num_out=conv4_num_out,
										stride=(1,2,2) if i==1 else (1,1,1),
										g=groups,
										first_block=(i==1))) for i in range(1,k_sec[4]+1)
					]))

		# conv5 - x7 (x8)
		num_mid *= 2
		conv5_num_out = 2 * conv4_num_out
		self.conv5 = nn.Sequential(OrderedDict([
					("B%02d"%i, MF_UNIT(num_in=conv4_num_out if i==1 else conv5_num_out,
										num_mid=num_mid,
										num_out=conv5_num_out,
										stride=(1,2,2) if i==1 else (1,1,1),
										g=groups,
										first_block=(i==1))) for i in range(1,k_sec[5]+1)
					]))

		# Define motion extractor after conv4
		self.motion_exctractor = Motion_Exctractor_MEAN( inplanes=conv4_num_out, num_embedding=14)

		# final
		self.tail = nn.Sequential(OrderedDict([
					('bn', nn.BatchNorm3d(conv5_num_out)),
					('relu', nn.ReLU(inplace=True))
					]))

		# self.globalpool = nn.Sequential(OrderedDict([
						# ('avg', nn.AvgPool3d(kernel_size=(8,7,7),  stride=(1,1,1))),
						# ('dropout', nn.Dropout(p=0.5)), only for fine-tuning
						# ]))
		# self.classifier = nn.Linear(conv5_num_out, num_classes)

		# Position related Linear Layers (input one-hot Dec-5)

		self.emb_prepool = nn.Sequential(OrderedDict([
						('avg', nn.AvgPool3d(kernel_size=(1,7,7),  stride=(1,1,1))),
						# ('dropout', nn.Dropout(p=0.5)), only for fine-tuning
						]))
		self.emb_postpool = nn.AvgPool1d(kernel_size=8)
		self.classifier = nn.Linear(conv5_num_out, num_classes)

		#############
		# Initialization
		initializer.xavier(net=self)

		if pretrained:
			pretrained_model=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'pretrained/MFNet2D_ImageNet1k-0000.pth')
			logging.info("Network:: graph initialized, loading pretrained model: `{}'".format(pretrained_model))
			assert os.path.exists(pretrained_model), "cannot locate: `{}'".format(pretrained_model)
			state_dict_2d = torch.load(pretrained_model)
			initializer.init_3d_from_2d_dict(net=self, state_dict=state_dict_2d, method='inflation')
		else:
			logging.info("Network:: graph initialized, use random inilization!")

	def forward(self, x):
		assert x.shape[2] == 16

		h = self.conv1(x)   # x224 -> x112
#		print(h.shape)
		h = self.maxpool(h) # x112 ->  x56
#		print(h.shape)

		h = self.conv2(h)   #  x56 ->  x56
#		print(h.shape)
		h = self.conv3(h)   #  x56 ->  x28
#		print(h.shape)
		h = self.conv4(h)   #  x28 ->  x14
#		print(h.shape)

		motion_h = self.motion_exctractor(h)
#		print(motion_h.shape)
		h = h + motion_h
#		print(h.shape)

		h = self.conv5(h)   #  x14 ->   x7
#		print(h.shape)
		
		# Dec 5: Use two one-hot or one-hot like embedding to represent forward and backward position

		h = self.tail(h)
#		print(h.shape)
		# h = self.globalpool(h)
		h = self.emb_prepool(h).squeeze(3).squeeze(3)
#		print(h.shape)
		h = self.emb_postpool(h)
#		print(h.shape)

		h = h.view(h.shape[0], -1)
#		print(h.shape)
		h = self.classifier(h)
#		print(h.shape)

		return h, motion_h

if __name__ == "__main__":
	import torch
	logging.getLogger().setLevel(logging.DEBUG)
	# ---------
	net = MFNET_SP_LINEAR_DUAL(num_classes=100, pretrained=False)
	data = torch.autograd.Variable(torch.randn(5,3,16,224,224))
	data = data#.cuda()
	net = net#.cuda()
	output, motion_h = net(data)
	# torch.save({'state_dict': net.state_dict()}, './tmp.pth')
	print (output.shape)