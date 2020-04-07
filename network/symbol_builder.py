# 将你的网络在这里进行import，从主程序输入的名称最好是全大写
import logging

from .c2dres50 import c2d50
from .mfnet_base import MFNET_BASE
from .mfnet_sp_linear_many_method1 import MFNET_SP_LINEAR_DUAL
from .mfnet_sp_linear_many_method2 import MFNET_SP_LINEAR
from .mfnet_sp_linear_5pixel_meth2_a import MFNET_FIVEP_LINEAR_PIXEL
from .mfnet_sp_linear_5pixel_meth2_b import MFNET_FIVEP_LINEAR_FRAME
from .mfnet_sp_linear_5pixel_meth2_ab import MFNET_FIVEP_LINEAR_FRA_PIX
from .mfnet_sp_linear_3pixel_m2_b import MFNET_THREEP_LINEAR_FRAME
from .mfnet_sp_linear_7pixel_m2_b import MFNET_SEVENP_LINEAR_FRAME
from .mfnet_sp_linear_5pixel_m2_b_linear15 import MFNET_FIVEP_LINEAR15_FRAME
from .mfnet_sp_linear_5pixel_m2_b_linear35 import MFNET_FIVEP_LINEAR35_FRAME
from .mfnet_sp_linear_5pixel_m2_b_conv3 import MFNET_FIVEP_LINEAR_FRAME_CONV3
from .mfnet_sp_linear_5pixel_m2_b_conv5 import MFNET_FIVEP_LINEAR_FRAME_CONV5



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
	elif name.upper() == "CHANGE_3B":
		net = MFNET_THREEP_LINEAR_FRAME(**kwargs)
	elif name.upper() == "CHANGE_7B":
		net = MFNET_SEVENP_LINEAR_FRAME(**kwargs)
	elif name.upper() == "CHANGE_5B15":
		net = MFNET_FIVEP_LINEAR15_FRAME(**kwargs)
	elif name.upper() == "CHANGE_5B35":
		net = MFNET_FIVEP_LINEAR35_FRAME(**kwargs)
	elif name.upper() == "CHANGE_5B_3":
		net = MFNET_FIVEP_LINEAR_FRAME_CONV3(**kwargs)
	elif name.upper() == "CHANGE_5B_5":
		net = MFNET_FIVEP_LINEAR_FRAME_CONV5(**kwargs)

	else:
		logging.error("network '{}'' not implemented".format(name))
		raise NotImplementedError()

	if print_net:
		logging.debug("Symbol:: Network Architecture:")
		logging.debug(net)

	input_conf = get_config(name, **kwargs)
	return net, input_conf

