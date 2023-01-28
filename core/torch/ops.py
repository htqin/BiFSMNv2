# -*- coding: utf-8 -*-
from abc import abstractmethod
from copy import deepcopy
from byteslim.core.registry import Registry

OPS = Registry('operators_for_slim')
QAT_OPS = Registry('operators_support_QAT')
Prune_OPS = Registry('operators_support_StructurePruning')
SVD_OPS = Registry('operators_support_StructurePruning')


class OpBase(object):
    """
    a base class for op attributes and configuration
    """
    def __init__(self):
        super(OpBase, self).__init__()

    def params(self):
        return self.params_name

    def quant_params(self):
        return [self.params_name[index] for index in self.quant_param_index]

    def quant_inputs(self):
        return [self.inputs_name[index] for index in self.quant_input_index]

    def process(self, module):
        # for weight_norm prune
        module_params_name = [name for name, _ in module.named_parameters()]
        for name in self.params_name:
            if name not in module_params_name and name + '_v' in module_params_name:
                self.prune_index[name + '_v'] = self.prune_index[name]
                self.prune_index.pop(name)


# torch modules
@OPS.register
@QAT_OPS.register
@Prune_OPS.register
@SVD_OPS.register
class Linear(OpBase):
    def __init__(self):
        super(Linear, self).__init__()
        self.inputs_name = ['input']
        self.params_name = ['weight', 'bias']
        self.quant_input_index = [0]
        self.quant_param_index = [0]
        self.quant_channel = {'params': [0]}
        self.svd_param_index = [0]
        self.prune_index = {'weight': 0}


@OPS.register
@QAT_OPS.register
@Prune_OPS.register
class Conv1d(OpBase):
    def __init__(self):
        super(Conv1d, self).__init__()
        self.inputs_name = ['input']
        self.params_name = ['weight', 'bias']
        self.quant_input_index = [0]
        self.quant_param_index = [0]
        self.quant_channel = {'params': [0]}
        self.prune_index = {'weight': 0}


@OPS.register
@QAT_OPS.register
@Prune_OPS.register
class Conv2d(OpBase):
    def __init__(self):
        super(Conv2d, self).__init__()
        self.inputs_name = ['input']
        self.params_name = ['weight', 'bias']
        self.quant_input_index = [0]
        self.quant_param_index = [0]
        self.quant_channel = {'params': [0]}
        self.prune_index = {'weight': 0}


@OPS.register
@QAT_OPS.register
@Prune_OPS.register
class ConvTranspose1d(OpBase):
    def __init__(self):
        super(ConvTranspose1d, self).__init__()
        self.inputs_name = ['input']
        self.params_name = ['weight', 'bias']
        self.quant_input_index = [0]
        self.quant_param_index = [0]
        self.quant_channel = {'params': [0]}
        self.prune_index = {'weight': 1}


@OPS.register
@QAT_OPS.register
@SVD_OPS.register
class LSTM(OpBase):
    def __init__(self):
        super(LSTM, self).__init__()
        self.inputs_name = ['input']
        self.params_name = [
            'weight_ih_l', 'weight_hh_l', 'bias_ih_l', 'bias_hh_l'
        ]
        self.quant_input_index = [0]
        self.quant_param_index = [0, 1]
        self.svd_param_index = [0, 1]
        self.quant_channel = {'params': [0, 0]}

    def process(self, module):
        params_dict = dict(module.named_parameters())
        self.params_name = []
        self.quant_param_index = []
        self.svd_param_index = []
        self.quant_channel['params'] = []
        for i, name in enumerate(params_dict.keys()):
            self.params_name.append(name)
            self.quant_channel['params'].append(0)
            if 'weight_' in name:
                self.quant_param_index.append(i)
                self.svd_param_index.append(i)


@OPS.register
@QAT_OPS.register
@SVD_OPS.register
class GRU(OpBase):
    def __init__(self):
        super(GRU, self).__init__()
        self.inputs_name = ['input']
        self.params_name = ['weight_ih_l', 'weight_hh_l']
        self.quant_input_index = [0]
        self.quant_param_index = [0, 1]
        self.svd_param_index = [0, 1]
        self.quant_channel = {'params': [0, 0]}

    def process(self, module):
        params_dict = dict(module.named_parameters())
        self.params_name = []
        self.quant_apram_index = []
        self.svd_param_index = []
        self.quant_channel['params'] = []
        for i, name in enumerate(params_dict.keys()):
            self.params_name.append(name)
            self.quant_channel['params'].append(0)
            if 'weight_' in name:
                self.quant_param_index.append(i)
                self.svd_param_index.append(i)


@OPS.register
@QAT_OPS.register
class CustomBMM(OpBase):
    def __init__(self):
        super(CustomBMM, self).__init__()
        self.inputs_name = ['q', 'k']
        self.params_name = []
        self.quant_input_index = [0, 1]
        self.quant_param_index = []
        self.quant_channel = []
