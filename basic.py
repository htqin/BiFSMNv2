import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from torch.nn import modules, Parameter
from torch.autograd import Function, Variable
# from pytorch_quantization import tensor_quant

activations = {
    'ReLU': nn.ReLU,
    'Hardtanh': nn.Hardtanh
}

def linear_quantize(input, bits=2):
    assert bits > 1, bits
    delta = torch.max(torch.abs(input)) / math.pow(2.0, bits-1)
    bound = math.pow(2.0, bits-1)
    min_val = - bound
    max_val = bound - 1
    rounded = torch.floor(input / delta + 0.5)
    clipped_value = torch.clamp(rounded, min_val, max_val) * delta
    return clipped_value

class BinaryQuantize(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = torch.sign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors
        grad_input = grad_output
        grad_input[input[0].gt(1)] = 0
        grad_input[input[0].lt(-1)] = 0
        return grad_input

class BinaryQuantize_Vanilla(Function):
    @staticmethod
    def forward(ctx, input, scale):
        ctx.save_for_backward(input)
        out = torch.sign(input)
        if scale != None:
            out = out * scale
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors
        grad_input = grad_output
        grad_input[input[0].gt(1)] = 0
        grad_input[input[0].lt(-1)] = 0
        return grad_input, None

class MyFunction(Function):
    @staticmethod
    def forward(ctx, input, scale):
        ctx.save_for_backward(input)
        out = input
        if scale != None:
            out = out * scale
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors
        grad_input = grad_output
        return grad_input, None

class BinaryQuantize_LCR(Function):
    @staticmethod
    #jingtaifangfa make the function can be directly used by .forward(), without instance.
    def forward(ctx, input, k, t):
        ctx.save_for_backward(input, k, t)
        out = torch.sign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, k, t = ctx.saved_tensors
        grad_input = k * t * (1 - torch.pow(torch.tanh(input * t), 2)) * grad_output
        return grad_input, None, None


class LPBQuantize(torch.nn.Module):
    def __init__(self):
        super(LPBQuantize, self).__init__()
        self.alpha = nn.Parameter(data = torch.tensor(1).float(), requires_grad=True)
        self.method = 'STE'
        # self.method = 'bireal'

    def forward(self, x, alpha=1):
        if self.method == 'STE':
            x = x * self.alpha
            binary_input_no_grad = MyFunction().apply(torch.sign(x), alpha)
            cliped_input = torch.clamp(x, -1.0, 1.0)
            x = binary_input_no_grad.detach() - cliped_input.detach() + cliped_input
        else:
            x = x * self.alpha
            out_forward = MyFunction().apply(torch.sign(x), alpha)
            mask1 = x < -1
            mask2 = x < 0
            mask3 = x < 1
            out1 = (-1) * mask1.type(torch.float32) + (x*x + 2*x) * (1-mask1.type(torch.float32))
            out2 = out1 * mask2.type(torch.float32) + (-x*x + 2*x) * (1-mask2.type(torch.float32))
            out3 = out2 * mask3.type(torch.float32) + 1 * (1- mask3.type(torch.float32))
            x = out_forward.detach() - out3.detach() + out3
        return x


class BiLinearLPB(torch.nn.Linear):
    def __init__(self, in_channels, out_channels, bias=True):
        super(BiLinearLPB, self).__init__(in_channels, out_channels, bias=bias)
        self.sw = None
        self.channel_threshold = nn.Parameter(torch.rand(1, in_channels) * 0.001, requires_grad=True)
        self.channel_threshold_1 = nn.Parameter(torch.rand(1, in_channels) * 0.001, requires_grad=True)
        self.channel_threshold_2 = nn.Parameter(torch.rand(1, in_channels) * 0.001, requires_grad=True)
        self.quant1 = LPBQuantize()
        self.quant2 = LPBQuantize()
        self.quant3 = LPBQuantize()

    def forward(self, input):
        ba1 = input - self.channel_threshold
        # ba1 = BinaryQuantize().apply(ba1)
        ba1 = self.quant1(ba1)

        ba2 = input - ba1 - self.channel_threshold_1
        sa2 = ba2.abs().mean(-1).unsqueeze(-1).detach()
        ba2 = self.quant2(ba2, sa2)
        ba = ba1 + ba2 - self.channel_threshold_2

        bw = self.weight
        # if self.sw == None:
        #     self.sw = nn.Parameter(data = self.weight.abs().mean(-1).view(-1, 1))
        # bw = self.quant3(bw) * self.sw
        sw = bw.abs().mean(-1).view(-1, 1).detach()
        bw = self.quant3(bw, sw)
        
        
        output = F.linear(ba, bw, self.bias)
        
        return output


class BiConv1dLPB(torch.nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):
        super(BiConv1dLPB, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias, padding_mode)
        self.sw = None
        self.channel_threshold = nn.Parameter(torch.rand(1, in_channels, 1) * 0.001, requires_grad=True)
        self.channel_threshold_1 = nn.Parameter(torch.rand(1, in_channels, 1) * 0.001, requires_grad=True)
        self.channel_threshold_2 = nn.Parameter(torch.rand(1, in_channels, 1) * 0.001, requires_grad=True)
        self.quant1 = LPBQuantize()
        self.quant2 = LPBQuantize()
        self.quant3 = LPBQuantize()

    def forward(self, input):
        ba1 = input - self.channel_threshold
        ba1 = self.quant1(ba1)

        ba2 = input - ba1 - self.channel_threshold_1
        sa2 = ba2.abs().view(ba2.size(0), ba2.size(1), -1).mean(-1).view(ba2.size(0), ba2.size(1), 1).detach()
        # ba2 = BinaryQuantize().apply(ba2) * sa2
        ba2 = self.quant2(ba2, sa2)
        ba = ba1 + ba2 - self.channel_threshold_2

        bw = self.weight
        # if self.sw == None:
        #     self.sw = nn.Parameter(data = bw.abs().view(bw.size(0), bw.size(1), -1).mean(-1).view(bw.size(0), bw.size(1), 1))
        # bw = self.quant3(bw) * self.sw
        sw = bw.abs().view(bw.size(0), bw.size(1), -1).mean(-1).view(bw.size(0), bw.size(1), 1).detach()
        bw = self.quant3(bw, sw)
        
        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[0] + 1) // 2, self.padding[0] // 2)
            output = F.conv1d(F.pad(ba, expanded_padding, mode='circular'),
                            bw, self.bias, self.stride,
                            _single(0), self.dilation, self.groups)
        output = F.conv1d(ba, bw, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
        
        return output

class BiConv2dLPB(torch.nn.Conv2d):
    # input: BiReal
    # weight: learnable_scale * 2; bw1 = self.weight; bw2 = 0.
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):
        super(BiConv2dLPB, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias, padding_mode)
        
        self.channel_threshold = nn.Parameter(torch.rand(1, in_channels, 1, 1) * 0.001, requires_grad=True)
        self.channel_threshold_1 = nn.Parameter(torch.rand(1, in_channels, 1, 1) * 0.001, requires_grad=True)
        self.channel_threshold_2 = nn.Parameter(torch.rand(1, in_channels, 1, 1) * 0.001, requires_grad=True)
        self.sw = None
        self.quant1 = LPBQuantize()
        self.quant2 = LPBQuantize()
        self.quant3 = LPBQuantize()

    def forward(self, input):
        ba1 = input - self.channel_threshold
        ba1 = self.quant1(ba1)

        ba2 = input - ba1 - self.channel_threshold_1
        sa2 = ba2.abs().mean(-1).unsqueeze(-1).detach()
        ba2 = self.quant2(ba2, sa2)
        ba = ba1 + ba2 - self.channel_threshold_2

        bw = self.weight
        # if self.sw == None:
        #     self.sw = nn.Parameter(data = bw.abs().view(bw.size(0), bw.size(1), -1).mean(-1).view(bw.size(0), bw.size(1), 1, 1))
        # bw = self.quant3(bw) * self.sw
        sw = bw.abs().mean(-1).unsqueeze(-1).detach()
        bw = self.quant3(bw, sw)
        
        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[0] + 1) // 2, self.padding[0] // 2)
            output = F.conv2d(F.pad(ba, expanded_padding, mode='circular'),
                            bw, self.bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        output = F.conv2d(ba, bw, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
        
        return output

biLinears = {
    False: nn.Linear,
    'lpb': BiLinearLPB,
}

biConv1ds = {
    False: nn.Conv1d,
    'lpb': BiConv1dLPB,
}

biConv2ds = {
    False: nn.Conv2d,
    'lpb': BiConv2dLPB,
}

def Count(module: nn.Module, id = -1):
    id = 0 if id == -1 else id
    for name, child_module in module.named_children():
        if isinstance(child_module, nn.ModuleList):
            for child_child_module in child_module:
                id = Count(child_child_module, id)
        else:
            id = Count(child_module, id)
            if isinstance(child_module, nn.Linear):
                id += 1
            elif isinstance(child_module, nn.Conv1d):
                id += 1
            elif isinstance(child_module, nn.Conv2d):
                id += 1
    return id

def Modify(module: nn.Module, bits=1, method='Sign', id=-1, first=-1, last=-1):
    id = 0 if id == -1 else id
    if method != False:
        for name, child_module in module.named_children():
            if isinstance(child_module, nn.ModuleList):
                for child_child_module in child_module:
                    _, id = Modify(child_child_module, bits=bits, method=method, id=id, first=first, last=last)
            else:
                _, id = Modify(child_module, bits=bits, method=method, id=id, first=first, last=last)
                if isinstance(child_module, nn.Linear):
                    id += 1
                    if id == first or id == last:
                        continue
                    if bits == 1:
                        new_layer = biLinears[method](child_module.in_features,
                                                            child_module.out_features,
                                                            False if child_module.bias == None else True)
                        new_layer.weight = module._modules[name].weight
                        new_layer.bias = module._modules[name].bias
                        if method == 'fate':
                            new_layer.init_sw()
                        module._modules[name] = new_layer
                        
                elif isinstance(child_module, nn.Conv1d):
                    id += 1
                    if id == first or id == last:
                        continue
                    if bits == 1:
                        new_layer = biConv1ds[method](in_channels=child_module.in_channels,
                                                            out_channels=child_module.out_channels,
                                                            kernel_size=child_module.kernel_size,
                                                            stride=child_module.stride,
                                                            padding=child_module.padding,
                                                            dilation=child_module.dilation,
                                                            groups=child_module.groups,
                                                            bias=False if child_module.bias == None else True,
                                                            padding_mode=child_module.padding_mode)
                        new_layer.weight = module._modules[name].weight
                        new_layer.bias = module._modules[name].bias
                        if method == 'fate':
                            new_layer.init_sw()
                        module._modules[name] = new_layer
                elif isinstance(child_module, nn.Conv2d):
                    id += 1
                    if id == first or id == last:
                        continue
                    if bits == 1:
                        new_layer = biConv2ds[method](in_channels=child_module.in_channels,
                                                            out_channels=child_module.out_channels,
                                                            kernel_size=child_module.kernel_size,
                                                            stride=child_module.stride,
                                                            padding=child_module.padding,
                                                            dilation=child_module.dilation,
                                                            groups=child_module.groups,
                                                            bias=False if child_module.bias == None else True,
                                                            padding_mode=child_module.padding_mode)
                        new_layer.weight = module._modules[name].weight
                        new_layer.bias = module._modules[name].bias
                        if method == 'fate':
                            new_layer.init_sw()
                        module._modules[name] = new_layer
                
    return module, id

def _get_attr(model, attr):
    res = []
    if hasattr(model, attr):
        tmp = getattr(model, attr)
        if isinstance(tmp, list):
            res.extend(tmp)
        else:
            res.append(tmp)
    for layer in model.children():
        res.extend(_get_attr(layer, attr))
    return res


