"""
Author: Yuecong Xu
"""
import logging
import os
from collections import OrderedDict
import torch
import torch.nn as nn

try:
	from . import initializer
except:
	import initializer


class BottleneckC2D(nn.Module):
	expansion = 4

	def __init__(self, inplanes, planes, stride=1, downsample=None):
		super(BottleneckC2D, self).__init__()
		self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
		self.bn1 = nn.BatchNorm3d(planes)
		self.conv2 = nn.Conv3d(planes, planes, kernel_size=(1, 3, 3), 
			stride=(1, stride, stride), padding=(0, 1, 1), bias=False)
		self.bn2 = nn.BatchNorm3d(planes)
		self.conv3 = nn.Conv3d(planes, planes * self.expansion, kernel_size=1, bias=False)
		self.bn3 = nn.BatchNorm3d(planes * self.expansion)
		self.relu = nn.ReLU(inplace=True)
		self.downsample = downsample
		self.stride = stride

	def forward(self, x):
		residual = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)
		out = self.relu(out)

		out = self.conv3(out)
		out = self.bn3(out)

		if self.downsample is not None:
			residual = self.downsample(x)

		out += residual
		out = self.relu(out)

		return out


class C2D(nn.Module):

	def __init__(self, block, layers, num_classes):
		self.inplanes = 64
		super(C2D, self).__init__()
		self.conv1 = nn.Conv3d(3, 64, kernel_size=(1, 7, 7), stride=2, padding=(0, 3, 3), bias=False)
		self.bn1 = nn.BatchNorm3d(64)
		self.relu = nn.ReLU(inplace=True)
		self.maxpool1 = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
		self.layer1 = self._make_layer(block, 64, layers[0])
		self.maxpool2 = nn.MaxPool3d(kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0))
		self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
		self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
		self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
		self.avgpool = nn.AdaptiveAvgPool3d(1)
		self.dropout = nn.Dropout(0.5)
		self.fc = nn.Linear(512 * block.expansion, num_classes)

		for name, m in self.named_modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)

	def _make_layer(self, block, planes, blocks, stride=1):
		downsample = None
		if stride != 1 or self.inplanes != planes * block.expansion:
			downsample = nn.Sequential(
				nn.Conv3d(self.inplanes, planes * block.expansion, 
					kernel_size=1, stride=(1, stride, stride), bias=False),
				nn.BatchNorm3d(planes * block.expansion),
			)

		layers = []
		layers.append(block(self.inplanes, planes, stride, downsample))
		self.inplanes = planes * block.expansion
		for i in range(1, blocks):
			layers.append(block(self.inplanes, planes))

		return nn.Sequential(*layers)

	def forward(self, x):
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.maxpool1(x)

		x = self.layer1(x)
		x = self.maxpool2(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)

		x = self.avgpool(x)
		x = x.view(x.size(0), -1)
		x = self.dropout(x)
		x = self.fc(x)

		return x


def c2d50(pretrained=False, num_classes=1000, **kwargs):
	"""Constructs a C2D ResNet-50 model. """
	model = C2D(BottleneckC2D, [3, 4, 6, 3], num_classes=num_classes, **kwargs)
	if pretrained:
		logging.info("Loading weight from imagenet and inflated")
		# print("Loading weight from imagenet and inflated")
		pretrained_model=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'pretrained/resnet50-19c8e357.pth')
		_pretrained = torch.load(pretrained_model)
		initializer.init_3d_from_2d_dict(model, _pretrained, method='inflation', contains_nl=False)
	return model


if __name__=='__main__':
	import torch
	img = torch.randn(1,3,32,224,224)
	net = c2d50(pretrained=True, num_classes=101)
	out = net(img)
	print(out.size())
