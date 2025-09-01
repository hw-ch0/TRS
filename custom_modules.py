import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

__all__ = ['QConv']

class STE_round(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_in):
        x_out = torch.round(x_in)
        return x_out
    @staticmethod
    def backward(ctx, g):
        return g

# ref. https://github.com/hustzxd/LSQuantization/blob/master/lsq.py
class QConv(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, args, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(QConv, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.STE_round = STE_round.apply
        self.bit_weight = args.bit_weight
        self.bit_act = args.bit_act

        if args.symmetric_fxp:
            self.Qn_w = -2**(self.bit_weight-1) + 1 # odd number of levels are used, does not used for 1-bit
        else:
            self.Qn_w = -2**(self.bit_weight-1) # does not used for 1-bit
        self.Qp_w = 2**(self.bit_weight-1) - 1 # does not used for 1-bit
        self.Qn_a = 0
        self.Qp_a = 2**(self.bit_act) - 1
        
        if args.baseline:
            self.sW = nn.Parameter(data = torch.tensor(1).float())
        else:
            self.register_buffer('sW', torch.tensor(1).float()) # fix sW when our optimizer is used
        self.sA = nn.Parameter(data = torch.tensor(1).float())

        self.register_buffer('init', torch.tensor([1]))
        self.register_buffer('prev_weight', torch.zeros_like(self.weight))
        self.register_buffer('prev_Qweight', torch.zeros_like(self.weight))
        self.register_buffer('MD2TP', torch.tensor(0).float())

    def weight_quantization(self, weight):
        if self.bit_weight == 32:
            return weight
        elif self.bit_weight == 1:
            weight = weight / (torch.abs(self.sW)+1e-6)
            weight = weight.clamp(-1, 1) # [-1, 1]
            weight = (weight + 1)/2 # [0, 1]
            weight_q = self.STE_round(weight) # {0, 1}
            if self.training:
                D2TP = 1 - (weight-weight_q).abs()
                self.MD2TP = D2TP.mean().detach().clone()
            weight_q = weight_q * 2 - 1 # {-1, 1}
            return weight_q
        else:
            weight = weight / (torch.abs(self.sW)+1e-6) # normalized such that 99% of weights lie in [-1, 1]
            weight = weight * 2**(self.bit_weight-1)
            weight = weight.clamp(self.Qn_w, self.Qp_w)
            weight_q = self.STE_round(weight) # {-2^(b-1), ..., 2^(b-1)-1}
            if self.training:
                D2TP = 0.5 - (weight-weight_q).abs()
                self.MD2TP = D2TP.mean().detach().clone()
            weight_q = weight_q / 2**(self.bit_weight-1) # fixed point representation
            return weight_q

    def act_quantization(self, x):
        if self.bit_act == 32:
            return x
        elif self.bit_act == 1:  # <-- 여기 수정
            x = x / (torch.abs(self.sA)+1e-6)
            x = x.clamp(0, 1) # [0, 1]
            x = self.STE_round(x) # {0, 1}
            return x
        else:
            x = x / (torch.abs(self.sA)+1e-6) # normalized such that 99% of activations lie in [0, 1]
            x = x * 2**self.bit_act
            x = x.clamp(self.Qn_a, self.Qp_a) # [0, 2^b-1]
            x = self.STE_round(x) # {0, ..., 2^b-1}
            x = x / 2**self.bit_act # fixed point representation
            return x

    def initialize(self, x):
        with torch.no_grad():
            self.sW.fill_(self.weight.std() * 3.0)
            self.sA.fill_(x.std() / math.sqrt(1 - 2/math.pi) * 3.0)
            self.prev_weight = self.weight.detach().clone()
            self.init.fill_(0)

        
    def forward(self, x):
        if self.training and self.init.item() == 1:
           self.initialize(x)
        
        Qweight = self.weight_quantization(self.weight)
        if self.training and self.bit_weight != 32:
            transition = (Qweight != self.prev_Qweight).float()
            setattr(self.weight, "transition", transition)
            self.prev_Qweight = Qweight.detach().clone()
        Qact = self.act_quantization(x)
        output = F.conv2d(Qact, Qweight, self.bias,  self.stride, self.padding, self.dilation, self.groups)

        return output
    
    def forward_c(self, x):
        if self.training and self.init.item() == 1:
           self.initialize(x)
        
        Qweight = self.weight_quantization(self.weight)
        if self.training and self.bit_weight != 32:
            transition = (Qweight != self.prev_Qweight).float()
            setattr(self.weight, "transition", transition)
            self.prev_Qweight = Qweight.detach().clone()
        Qact = self.act_quantization(x)
        output = F.conv2d(Qact, Qweight, self.bias,  self.stride, self.padding, self.dilation, self.groups)

        return output