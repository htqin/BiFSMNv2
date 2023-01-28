import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import modules, Parameter
import torch.nn.functional as F
from typing import Union, List

class FSMN_block(nn.Module):
    def __init__(self, F1, F3, F2, 
                 context: int = 3,
                 dilation: int = 1):
        super(FSMN_block, self).__init__()
        self.fc1 = nn.Linear(F1, F3)
        self.fc2 = nn.Linear(F2, F3, bias=False)
        self.conv1d = nn.Conv1d(F1, F2, kernel_size=2*context+1,
                                  dilation=dilation,
                                  groups=F1,
                                  padding=context,
                                  bias=False)
        self.act_func = nn.PReLU()
        # self.channel_drop = nn.Dropout2d(p=0.05)

    def forward(self, inp):
        # return nn.ReLU(self.fc1(inp) + self.fc2(self.conv1d(inp.transpose(1, 2)).transpose(1, 2)))
        out = self.fc1(inp) + self.fc2(self.conv1d(inp.transpose(1, 2)).transpose(1, 2))
        # out = self.channel_drop(out)
        self.layer_output = out
        return self.act_func(out), out


class FSMN(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 out_features: int = 128,
                 proj_features: int = 256,
                 num_classes: int = 12,
                 num_layers: int = 8,
                 context: int = 3,
                 dilation: Union[List[int], int] = 1,
                 distill: bool = False,
                 **kwargs):
        super(FSMN, self).__init__()
        if isinstance(dilation, int):
            dilation = [dilation] * num_layers
        self.enc_layers = nn.ModuleList([
            FSMN_block(32 if i == 0 else out_features,
                 out_features,
                 proj_features,
                 context=context,
                 dilation=dilation[i]) for i in range(num_layers)
        ])
        self.distill = distill
        self.fc = nn.Linear(in_features=out_features*32, out_features=num_classes)

    
    def forward(self, x):
        x = x.squeeze(1).transpose(1, 2)
        features = []
        for fsmn in self.enc_layers:
            x, feat = fsmn(x)
            features.append(feat)
        x = x.view(x.shape[0], -1)
    
        return self.fc(x)

