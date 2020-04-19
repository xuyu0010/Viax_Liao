import torch
from thop import profile
from thop import clever_format
from torchsummary import summary

# logging.getLogger().setLevel(logging.DEBUG)
# ---------
net = MFNET_SP_LINEAR_REGION(num_classes=100, pretrained=False)
data = torch.autograd.Variable(torch.randn(1,3,16,224,224))
flops, params = profile(net, inputs=(data, ))
# summary(net, (3,16,224,224))
flops, params = clever_format([flops, params], "%.3f")
print("Number of flops: {}".format(flops))
print("Number of params: {}".format(params))