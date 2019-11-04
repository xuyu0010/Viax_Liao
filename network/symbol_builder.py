import logging

from .c2dres50 import c2d50

from .config import get_config

def get_symbol(name, print_net=False, **kwargs):

	if name.upper() == "C2D_50":
		net = c2d50(**kwargs)

	else:
		logging.error("network '{}'' not implemented".format(name))
		raise NotImplementedError()

	if print_net:
		logging.debug("Symbol:: Network Architecture:")
		logging.debug(net)

	input_conf = get_config(name, **kwargs)
	return net, input_conf

