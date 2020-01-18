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
	from . import CombinedPooling as CP
except:
	import CombinedPooling as CP


class BN_AC_CONV3D(nn.Module):

	def __init__(self, num_in, num_filter,
				 kernel=(1,1,1), pad=(0,0,0), stride=(1,1,1), g=1, bias=False):
		super(BN_AC_CONV3D, self).__init__()
		self.bn = nn.BatchNorm3d(num_in)
		self.relu = nn.ReLU(inplace=True)
		self.conv = nn.Conv3d(num_in, num_filter, kernel_size=kernel, padding=pad,
							   stride=stride, groups=g, bias=bias)

	def forward(self, x):
		h = self.relu(self.bn(x))
		h = self.conv(h)
		return h


class MF_UNIT(nn.Module):

	def __init__(self, num_in, num_mid, num_out, g=1, stride=(1,1,1), first_block=False, use_3d=True):
		super(MF_UNIT, self).__init__()
		num_ix = int(num_mid/4)
		kt,pt = (3,1) if use_3d else (1,0)
		# prepare input
		self.conv_i1 =     BN_AC_CONV3D(num_in=num_in,  num_filter=num_ix,  kernel=(1,1,1), pad=(0,0,0))
		self.conv_i2 =     BN_AC_CONV3D(num_in=num_ix,  num_filter=num_in,  kernel=(1,1,1), pad=(0,0,0))
		# main part
		self.conv_m1 =     BN_AC_CONV3D(num_in=num_in,  num_filter=num_mid, kernel=(kt,3,3), pad=(pt,1,1), stride=stride, g=g)
		if first_block:
			self.conv_m2 = BN_AC_CONV3D(num_in=num_mid, num_filter=num_out, kernel=(1,1,1), pad=(0,0,0))
		else:
			self.conv_m2 = BN_AC_CONV3D(num_in=num_mid, num_filter=num_out, kernel=(1,3,3), pad=(0,1,1), g=g)
		# adapter
		if first_block:
			self.conv_w1 = BN_AC_CONV3D(num_in=num_in,  num_filter=num_out, kernel=(1,1,1), pad=(0,0,0), stride=stride)

	def forward(self, x):

		h = self.conv_i1(x)
		x_in = x + self.conv_i2(h)

		h = self.conv_m1(x_in)
		h = self.conv_m2(h)

		if hasattr(self, 'conv_w1'):
			x = self.conv_w1(x)

		return h + x

class MFBPNET_3D_LATERAL_STABLE(nn.Module):

	def __init__(self, num_classes, pretrained=False, **kwargs):
		super(MFBPNET_3D_LATERAL_STABLE, self).__init__()

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

		# final
		self.tail = nn.Sequential(OrderedDict([
					('bn', nn.BatchNorm3d(conv5_num_out)),
					('relu', nn.ReLU(inplace=True))
					]))		
		
		self.globalpool = nn.Sequential(OrderedDict([
						('avg', nn.AvgPool3d(kernel_size=(8,7,7),  stride=(1,1,1))),
						# ('dropout', nn.Dropout(p=0.5)), only for fine-tuning
						]))

		self.concat_classifier = nn.Linear(2*conv5_num_out, num_classes) # This attempts to classify directly after concatenating two pooled features (Run ID 8, 9, 12, 13)
		"""
		The following would replace the original global pooling method with a combination of 
		compact bilinear pooling and average pooling with multiple linear layers
		"""
		self.cbp_out = 5 * conv5_num_out # change bilinear channels
		# Change between CP/CP_attn to import different file (stable/attn ver), CP_attn is used for Ablation Study ATN 4
		self.combinedpool = CP.CombinedPooling(num_in=conv5_num_out, num_out=self.cbp_out, 
					num_mid1=4*conv5_num_out, num_mid2=2*conv5_num_out, kernel_s=(1, 7, 7), kernel_t=(7, 1, 1), pad=0, stride=1)
		
		# Sigmoid for attention (Comment for Run ID < 54)
		self.sigmoid = nn.Sigmoid()

		#############
		# Initialization
		initializer.xavier(net=self)

		if pretrained:
			import torch
			load_method='inflation' # 'random', 'inflation'
			pretrained_model=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'pretrained/MFNet2D_ImageNet1k-0000.pth')
			logging.info("Network:: graph initialized, loading pretrained model: `{}'".format(pretrained_model))
			assert os.path.exists(pretrained_model), "cannot locate: `{}'".format(pretrained_model)
			state_dict_2d = torch.load(pretrained_model)
			initializer.init_3d_from_2d_dict(net=self, state_dict=state_dict_2d, method=load_method)
		else:
			logging.info("Network:: graph initialized, use random inilization!")

	def forward(self, x):
		assert x.shape[2] == 16

		h = self.conv1(x)   # x224 -> x112
		h = self.maxpool(h) # x112 ->  x56

		h = self.conv2(h)   #  x56 ->  x56
		h = self.conv3(h)   #  x56 ->  x28
		h = self.conv4(h)   #  x28 ->  x14
		h = self.conv5(h)   #  x14 ->   x7

		h = self.tail(h)
		
		g = h.clone()
		g = self.globalpool(g)
		g = g.view(h.shape[0], -1)
		h = self.combinedpool(h, num_out=self.cbp_out)

		h = torch.cat((g, h), 1)
		# Sigmoid follows concat (Comment for Run ID < 54 & Ablation studies)
		# sig = self.sigmoid(h)
		# h = torch.mul(h, sig)

		h = self.concat_classifier(h) # the change to concat the two pooled features lead to change in the classifier. 

		return h

if __name__ == "__main__":
	import torch
	logging.getLogger().setLevel(logging.DEBUG)
	# ---------
	net = MFBPNET_3D_LATERAL_STABLE(num_classes=100, pretrained=False)
	data = torch.autograd.Variable(torch.randn(1,3,16,224,224))
	output = net(data)
	# torch.save({'state_dict': net.state_dict()}, './tmp.pth')
	print (output.shape)