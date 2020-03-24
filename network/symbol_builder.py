# 将你的网络在这里进行import，从主程序输入的名称最好是全大写
import logging

from .c2dres50 import c2d50
from .mfnet_base import MFNET_BASE
from .mfnet_sp_linear_many_method1 import MFNET_SP_LINEAR_DUAL
from .mfnet_sp_linear_many_method2 import MFNET_SP_LINEAR
from .mfnet_sp_linear_5pixel_meth2_a import MFNET_FIVEP_LINEAR_PIXEL
from .mfnet_sp_linear_5pixel_meth2_b import MFNET_FIVEP_LINEAR_FRAME
from .mfnet_sp_linear_5pixel_meth2_ab import MFNET_FIVEP_LINEAR_FRA_PIX


from .config import get_config

def get_symbol(name, print_net=False, **kwargs):

	if name.upper() == "C2D_50":
		net = c2d50(**kwargs)
	elif name.upper() == "MFNET_BASE":
		net = MFNET_BASE(**kwargs)
	elif name.upper() == "CHANGE_1":
		net = MFNET_SP_LINEAR_DUAL(**kwargs)
	elif name.upper() == "CHANGE_2":
		net = MFNET_SP_LINEAR(**kwargs)
	elif name.upper() == "CHANGE_A":
		net = MFNET_FIVEP_LINEAR_PIXEL(**kwargs)
	elif name.upper() == "CHANGE_B":
		net = MFNET_FIVEP_LINEAR_FRAME(**kwargs)
	elif name.upper() == "CHANGE_AB":
		net = MFNET_FIVEP_LINEAR_FRA_PIX(**kwargs)

	else:
		logging.error("network '{}'' not implemented".format(name))
		raise NotImplementedError()

	if print_net:
		logging.debug("Symbol:: Network Architecture:")
		logging.debug(net)

	input_conf = get_config(name, **kwargs)
	return net, input_conf

