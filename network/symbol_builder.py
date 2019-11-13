# 将你的网络在这里进行import，从主程序输入的名称最好是全大写
import logging

from .c2dres50 import c2d50
from .mfnet_3d import MFNET_3D

from .config import get_config

def get_symbol(name, print_net=False, **kwargs):

	if name.upper() == "C2D_50":
		net = c2d50(**kwargs)
	if name.upper() == "MFNET_3D":
		net = MFNET_3D(**kwargs)

	else:
		logging.error("network '{}'' not implemented".format(name))
		raise NotImplementedError()

	if print_net:
		logging.debug("Symbol:: Network Architecture:")
		logging.debug(net)

	input_conf = get_config(name, **kwargs)
	return net, input_conf

