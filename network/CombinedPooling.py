"""
Combining Compact Bilinear Pooling with an Average Pooling
"""
import torch
from torch import nn
from torch.autograd import Variable
"""
This edition of Compact Bilinear Pooling is from https://github.com/gdlg/pytorch_compact_bilinear_pooling, with pytorch version of 030
"""
try:
	from . import compact_bilinear_pooling030 as CBP
except:
	import compact_bilinear_pooling030 as CBP


class CombinedPooling(nn.Module):

	def __init__(self, num_in, num_out, num_mid1, num_mid2, kernel_s, kernel_t, pad=0, stride=1):
		super(CombinedPooling, self).__init__()
		self.cbp = CBP.CompactBilinearPooling(num_in, num_in, num_out)
		self.relu = nn.ReLU(inplace=False)
		self.bn3d = nn.BatchNorm3d(num_out + num_in) # for single cbp as in Run ID 12
		concat_out = num_out + num_in # this is needed for slow-fast network inspired cbp
		self.s_ap = nn.AvgPool3d(kernel_size=kernel_s, stride=stride, padding=pad)
		self.t_ap = nn.AvgPool3d(kernel_size=kernel_t, stride=stride, padding=pad) 
		self.dual_ap = nn.AvgPool3d(kernel_size=(2, 1, 1), stride=(1, 1, 1), padding=0) # dual_ap is used to get the avg pooling for subsequent two frames, concatenate to bilinear pooled result
		"""
		The following linear reduction is as implemented in Runs with same struct as Run ID 47
		"""
		self.linear1 = nn.Linear(concat_out, num_mid1) # this is needed for slow-fast network inspired cbp
		self.linear2 = nn.Linear(num_mid1, num_mid2)
		self.linear3 = nn.Linear(num_mid2, num_in)

	def forward(self, x, num_out):
		"""
		The below lines are necessary steps to get the dual ap before concatenating with bilinear pooled feature
		This for slow-fast network inspired cbp
		"""
		dual = self.dual_ap(x) # {16, 768, 7, 7, 7}
		dual = dual.permute(2, 0, 3, 4, 1) # {7, 16, 7, 7, 768}

		""" 
		The below section refers to attempt for the new implementation of the first attempt using library by gdlg (Run ID 12)
		"""
		x = x.permute(2, 0, 3, 4, 1)
		h = Variable(torch.zeros(x.shape[0]-1, x.shape[1], x.shape[2], x.shape[3], num_out)).cuda()
		for t in range(x.shape[0]-1):
			h[t] = self.cbp(x[t], x[t+1])

		h = torch.cat((h, dual), 4)
		h = h.permute(1, 4, 0, 2, 3) # {16, concat_out, 7, 7, 7}

		h = self.bn3d(h.contiguous())
		h = self.s_ap(h)
		h = self.relu(h)
		h = self.t_ap(h)
		h = h.view(h.shape[0], -1)
		"""
		The following linear reduction is as implemented in Runs with same struct as Run ID 12
		"""
		h = self.relu(self.linear1(h))
		h = self.relu(self.linear2(h))
		h = self.linear3(h) 
		
		return h

if __name__ == '__main__':

	video_in = Variable(torch.randn(16, 768, 8, 7, 7)).cuda()

	layer = CombinedPooling(768, 6000, 3000, 1000, (1, 7, 7), (7, 1, 1))
	layer.cuda()
	layer.train()

	out = layer(video_in, num_out=6000)
	print(out.shape)