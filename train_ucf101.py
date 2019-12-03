import os    #os模块提供了多数操作系统的功能接口函数;
import json	#JSON (JavaScript Object Notation) 是一种轻量级的数据交换格式。
		#Python3 中可以使用 json 模块来对 JSON 数据进行编解码，
			#它主要提供了四个方法： dumps、dump、loads、load。
import socket	#套节字，信息交互
import logging	#logging模块是Python内置的标准模块，主要用于输出运行日志，可以设置输出日志的等级、日志保存路径、日志文件回滚等；
import argparse	#argparse是python标准库里面用来处理命令行参数的库
			#命令行参数分为位置参数和选项参数：

import torch	# 包含了多维张量的数据结构以及基于其上的多种数学操作。
import torch.nn.parallel	#并行计算
import torch.distributed as dist	#分布式Pyrorch允许您在多台机器之间交换Tensors。使用此软件包，您可以通过多台机器和更大的小批量扩展网络训练。

import dataset
from train_model import train_model
from network.symbol_builder import get_symbol	#导入get_symbol函数

torch.backends.cudnn.enabled = False	#禁用cudnn(cudnn使用非确定性算法)
# 参数输入
parser = argparse.ArgumentParser(description="PyTorch Video Classification Parser")
	#创建一个解析对象
# debug
	#向该对象中添加你要关注的命令行参数和选项
parser.add_argument('--debug-mode', type=bool, default=True,
					help="print all setting for debugging.")
# io
parser.add_argument('--dataset', default='UCF101', choices=['UCF101', 'HMDB51', 'Kinetics'],
					help="path to dataset")
parser.add_argument('--clip-length', default=32,
					help="define the length of each input sample.")
parser.add_argument('--train-frame-interval', type=int, default=2,
					help="define the sampling interval between frames.")
parser.add_argument('--val-frame-interval', type=int, default=2,
					help="define the sampling interval between frames.")
parser.add_argument('--task-name', type=str, default='',
					help="name of current task, leave it empty for using folder name")
parser.add_argument('--model-dir', type=str, default="./exps/models",
					help="set logging file.")
parser.add_argument('--log-file', type=str, default="",
					help="set logging file.")
# device
# parser.add_argument('--gpus', type=str, default="0,1,2,3,4,5,6,7",
					# help="define gpu id")
# algorithm
parser.add_argument('--network', type=str, default='MFNET_3D',
					help="choose the base network")
# initialization with priority (the next step will overwrite the previous step)
# - step 1: random initialize
# - step 2: load the 2D pretrained model if `pretrained_2d' is True
# - step 3: load the 3D pretrained model if `pretrained_3d' is defined
# - step 4: resume if `resume_epoch' >= 0
parser.add_argument('--pretrained_2d', type=bool, default=True,
					help="load default 2D pretrained model.")
parser.add_argument('--pretrained_3d', type=str, 
		    			# default=None,
		    			default='./network/pretrained/MFNet3D_Kinetics-400_72.8.pth'
					help="load default 3D pretrained model.")
parser.add_argument('--resume-epoch', type=int, default=-1,
					help="resume train")
# optimization
parser.add_argument('--fine-tune', type=bool, default=False,
					help="apply different learning rate for different layers")
parser.add_argument('--batch-size', type=int, default=64,
					help="batch size")
parser.add_argument('--lr-base', type=float, default=0.005,
					help="learning rate")
parser.add_argument('--lr-steps', type=list, default=[int(1e5*x) for x in [2, 4, 6, 7]],
					help="number of samples to pass before changing learning rate") # 1e6 million
parser.add_argument('--lr-factor', type=float, default=0.1,
					help="reduce the learning with factor")
parser.add_argument('--save-frequency', type=float, default=10,
					help="save once after N epochs")
parser.add_argument('--end-epoch', type=int, default=40,
					help="maxmium number of training epoch")
parser.add_argument('--random-seed', type=int, default=1,
					help='random seed (default: 1)')
# distributed training
parser.add_argument('--backend', default='nccl', type=str, choices=['gloo', 'nccl'],
					help='Name of the backend to use')
parser.add_argument('--world-size', default=1, type=int,
					help='number of distributed processes')
parser.add_argument('--dist-url', default='tcp://192.168.0.11:23456', type=str,
					help='url used to set up distributed training')
# 参数自动输入
def autofill(args):		#args: 要解析的命令行参数列表。
	# customized
	if not args.task_name:	#task_name ?
		args.task_name = os.path.basename(os.getcwd())	#os.getcwd()(get current work directory)，获取当前工作的目录
								#os.path.basename(path)——返回文件名
	if not args.log_file:	#log_file ?
		if os.path.exists("./exps/logs"):		#os.path.exists(path)——检验指定的对象是否存在。是True,否则False
			args.log_file = "./exps/logs/{}_at-{}.log".format(args.task_name, socket.gethostname())
		else:						#socket.gethostname()获取本机IP
								#format格式化函数
			args.log_file = ".{}_at-{}.log".format(args.task_name, socket.gethostname())
	# fixed 	model_prefix ?
	args.model_prefix = os.path.join(args.model_dir, args.task_name)	# os.path.join(path, name)—连接目录和文件名
	return args
# 记录文件创立
def set_logger(log_file='', debug_mode=False):
	if log_file:
		if not os.path.exists("./"+os.path.dirname(log_file)):
			os.makedirs("./"+os.path.dirname(log_file))	#os.mkdir(path)——创建path指定的目录，该参数不能省略。
									#注意：这样只能建立一层，要想递归建立可用：os.makedirs()
		handlers = [logging.FileHandler(log_file), logging.StreamHandler()]
	else:			#FileHandler将日志写入到文件, StreamHandler将日志同时输出到屏幕
		handlers = [logging.StreamHandler()]	#Handlers 处理程序将日志记录（由记录器创建）发送到适当的目标。

	""" add '%(filename)s:%(lineno)d %(levelname)s:' to format show source file """
			
	logging.basicConfig(level=logging.DEBUG if debug_mode else logging.INFO,	#设置基础配置	
				format='%(asctime)s: %(message)s',
				datefmt='%Y-%m-%d %H:%M:%S',
				handlers = handlers)

if __name__ == "__main__":

	# set args
	args = parser.parse_args()
	args = autofill(args)

	set_logger(log_file=args.log_file, debug_mode=args.debug_mode)
	logging.info("Using pytorch {} ({})".format(torch.__version__, torch.__path__))	#日志级别大小关系为：CRITICAL > ERROR > WARNING > INFO > DEBUG > NOTSET
	logging.info("Start training with args:\n" +
				 json.dumps(vars(args), indent=4, sort_keys=True))	#dumps对python对象进行序列化

	# set device states
	# os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus # before using torch
	# assert torch.cuda.is_available(), "CUDA is not available"
	torch.manual_seed(args.random_seed)			#为CPU设置种子用于生成随机数，以使得结果是确定的
	# torch.cuda.manual_seed(args.random_seed)		#为当前GPU设置随机种子；如果使用多个GPU，应该使用

	# distributed training # 多机训练才需要，默认是没用的
	args.distributed = args.world_size > 1
	if args.distributed:
		import re, socket
		rank = int(re.search('192.168.0.(.*)', socket.gethostname()).group(1))
		logging.info("Distributed Training (rank = {}), world_size = {}, backend = `{}'".format(
					 rank, args.world_size, args.backend))
		dist.init_process_group(backend=args.backend, init_method=args.dist_url, rank=rank,
								group_name=args.task_name, world_size=args.world_size)
			#初始化默认的分布式进程组，backend-后端使用；init_method指定如何初始化进程组的URL；world_size–参与作业的进程数；rank–当前流程的排名。
		
	# load dataset related configuration # 数据库导入
	dataset_cfg = dataset.get_config(name=args.dataset)

	# creat model with all parameters initialized # 创建网络和整合网络参数
	net, input_conf = get_symbol(name=args.network,
					 pretrained=args.pretrained_2d if args.resume_epoch < 0 else None,
					 print_net=True if args.distributed else False,
					 **dataset_cfg)

	# training # 在这里开始训练
	kwargs = {}
	kwargs.update(dataset_cfg)
	kwargs.update({'input_conf': input_conf})
	kwargs.update(vars(args))
	train_model(sym_net=net, **kwargs)
	
	#如果我们不确定要往函数中传入多少个参数，或者我们想往函数中以列表和元组的形式传参数时，那就使要用*args；
	#如果我们不知道要往函数中传入多少个关键词参数，或者想传入字典的值作为关键词参数时，那就要使用**kwargs。
	
	
