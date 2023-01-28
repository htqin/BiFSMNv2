import sys
import inspect
import torch
import torch.nn as nn
from abc import abstractmethod
from copy import deepcopy
from collections import OrderedDict
from byteslim.nas.torch.utils import ConverterMap
from byteslim.core.config import Logger, Verbose
from .graph_operation import replace_module


# OFA_config.config is class type wise and identified by instance.__class__.__name__
class OFAConfig(object):
    def __init__(self,
                 SearchSpace: dict,
                 stages: list = ['width'],
                 binding_layer: dict = {},
                 static_layer: list = []):
        self.SearchSpace = SearchSpace
        self.stages = stages
        self.binding_layer = binding_layer
        self.static_layer = static_layer


class Compressor(object):
    def __init__(self, model, config):
        self.config = config
        self.model = model
        self.preprocess(model)

    # Support input compress_info as {module_class: compress_info} format for easy use.
    # But it need be unfold before it can be used to compress model.
    def unfold_compress_info(self):
        for module_name, module in self.model.named_modules():
            op_type = module.__class__.__name__
            if not self.config.NeedSkip(module_name, op_type,
                                        framework='torch'):
                op_config = self.config.get_type_config(op_type)
                if hasattr(module,
                           'slim_config') and module.slim_config['processed']:
                    parent_module = self.get_related_SlimModule(module)
                    module_name = parent_module.slim_config['identical_name']
                else:
                    parent_module = module
                    self.config.set_layer_config(module_name, op_config)
                op_config.process(parent_module)
                setattr(op_config, 'op_name', module_name)
                self.config.set_layer_config(module_name, op_config)

    def get_module(self, name):
        for n, m in self.model.named_modules():
            if not hasattr('slim_config'):
                continue
            if m.slim_config['identical_name'] == name:
                return m

    def get_related_SlimModule(self, module):
        if isinstance(module, SlimModule) or not hasattr(module, 'slim_config'):
            return None
        assert module.slim_config['identical_name'].endswith('_origin'), 'The identical name of module name '\
            'should be end with "origin" but get {}'.format(module.slim_config['identical_name'])
        target_identical_name = module.slim_config['identical_name'][:-7]
        for m in self.model.modules():
            if hasattr(m, 'slim_config'):
                if m.slim_config['identical_name'] == target_identical_name:
                    return m
        return None

    # Should be used after unfold compress info.
    def get_layer_config(self, name):
        return self.config.get_layer_config(name)

    def get_module_identity(self, module):
        if hasattr(module, 'slim_config'):
            return module.slim_config['identical_name']
        else:
            return None

    # target_module should be a list of module class, such as nn.Linear, nn.Conv2d
    def preprocess(self, model: nn.Module):
        self.unfold_compress_info()
        target_names = list(self.config.layer_config.keys())

        for n, m in model.named_modules():
            config = {}
            is_base_model = len(list(m.children())) == 0
            config['is_base_model'] = is_base_model
            config['identical_name'] = n + '_origin'
            config['processed'] = False
            if n in target_names:
                if hasattr(m, 'slim_config'):
                    if m.slim_config['processed']:
                        continue
                config['processed'] = True
                slim_module = SlimModule(m)
                setattr(m, 'slim_config', config)
                slim_config = deepcopy(config)
                slim_config['identical_name'] = n
                setattr(slim_module, 'slim_config', slim_config)
                replace_module(model, n, slim_module)
            else:
                setattr(m, 'slim_config', config)

    @abstractmethod
    def apply(self, name, ratio):
        pass

    @abstractmethod
    def compress(self, model):
        pass


class Compress_Reward(object):
    def __init__(self, reduced_flops=0, reduced_params=0):
        self.reduced_flops = reduced_flops
        self.reduced_params = reduced_params

    @abstractmethod
    def __call__(self, module, input):
        pass


class SlimModule(nn.Module):
    def __init__(self, base_model):
        super(SlimModule, self).__init__()
        self.base_model = base_model
        self.base_params = dict(self.base_model.named_parameters())
        for param_name, param in self.base_params.items():
            delattr(self.base_model, param_name)
            self.register_parameter(param_name, param)
        self.init_params_tensor()
        self.assign_params_tensor()
        self.type = base_model.__class__.__name__

        self.pre_ops = nn.ModuleDict()
        self.post_ops = nn.ModuleDict()
        self.pre_ops_order = OrderedDict()
        self.post_ops_order = OrderedDict()
        self.original_forward = False

    def __setattr__(self, name: str, value) -> None:
        if hasattr(self, 'base_params') and isinstance(value, nn.Parameter):
            self.base_params[name] = value
        return super().__setattr__(name, value)

    def register_pre_op(self, module: nn.Module, name: str, priority: int):
        setattr(module, 'parent_model', self)
        self.pre_ops.update({name: module})
        self.pre_ops_order.update({name: priority})
        sorted_list = sorted(self.pre_ops_order.items(), key=lambda x: x[1])
        self.pre_ops_order = OrderedDict(sorted_list)
        Logger(Verbose.DEBUG)("Updated pre_ops Order is {}".format(
            self.pre_ops_order))

    def init_params_tensor(self):
        self.current_params = {}
        for n, p in self.base_params.items():
            #Convert params to a tensor
            if isinstance(p, nn.Parameter):
                self.current_params[n] = p.clone()

    def init_act_tensor(self, *args, **kwargs):
        sig = inspect.signature(self.base_model.forward)
        inputs = OrderedDict(sig.parameters)
        inputs_names = list(inputs.keys())
        inputs_dict = OrderedDict()
        for i, act in enumerate(args):
            inputs_dict[inputs_names[i]] = act
            inputs.pop(inputs_names[i])
        for k, v in kwargs.items():
            inputs_dict[k] = v
            inputs.pop(k)
        for k, v in inputs.items():
            if k not in inputs_dict.keys() and v.default is not inspect._empty:
                inputs_dict[k] = v.default
        return inputs_dict

    def assign_params_tensor(self):
        for name, param_tensor in self.current_params.items():
            setattr(self.base_model, name, param_tensor)

    def recover_base_model(self):
        for name, param in self.base_params.items():
            if hasattr(self.base_model, name):
                delattr(self.base_model, name)
            self.base_model.register_parameter(name, param)
        if hasattr(self.base_model, 'slim_config'):
            delattr(self.base_model, 'slim_config')

    def forward(self, *args, **kwargs):
        self.init_params_tensor()
        inputs = self.init_act_tensor(*args, **kwargs)
        if self.original_forward:
            return self.model(**inputs)
        else:
            for n, p in self.pre_ops_order.items():
                inputs = self.pre_ops[n](**inputs)
            self.assign_params_tensor()

            if 'flatten_parameters' in dir(self.base_model):
                self.base_model.flatten_parameters()

            inputs = self.base_model(**inputs)
            for n, p in self.post_ops_order.items():
                inputs = self.post_ops[n](**inputs)
            delattr(self, 'current_params')
            if isinstance(inputs, dict):
                return tuple(inputs.values())
            else:
                return inputs


class SlimWorker(nn.Module):
    def __init__(self):
        super(SlimWorker, self).__init__()

    def __repr__(self):
        return self.__class__.__name__

    def _get_target_param_name(self, param_name) -> list:
        if isinstance(self.parent_model.base_params[param_name], nn.Parameter):
            return [param_name]
        else:
            return [name for name in self.parent_model.base_params[param_name]]

    def get_param(self, param_name) -> torch.Tensor:
        if param_name in self.parent_model.current_params.keys():
            return self.parent_model.current_params[param_name]
        else:
            return None

    def assign_param(self, param_name, param_value):
        self.parent_model.current_params[param_name] = param_value

    def named_children(self):
        memo = set()
        for name, module in self._modules.items():
            if module is not None and module not in memo and name != 'parent_model':
                memo.add(module)
                yield name, module

    def __setattr__(self, name: str, value) -> None:
        def remove_from(*dicts_or_sets):
            for d in dicts_or_sets:
                if name in d:
                    if isinstance(d, dict):
                        del d[name]
                    else:
                        d.discard(name)

        params = self.__dict__.get('_parameters')
        if isinstance(value, nn.Parameter):
            if params is None:
                raise AttributeError(
                    "cannot assign parameters before Module.__init__() call")
            remove_from(self.__dict__, self._buffers, self._modules,
                        self._non_persistent_buffers_set)
            self.register_parameter(name, value)
        elif params is not None and name in params:
            if value is not None:
                raise TypeError("cannot assign '{}' as parameter '{}' "
                                "(torch.nn.Parameter or None expected)".format(
                                    torch.typename(value), name))
            self.register_parameter(name, value)
        else:
            modules = self.__dict__.get('_modules')
            if isinstance(value, nn.Module):
                if modules is None:
                    raise AttributeError(
                        "cannot assign module before Module.__init__() call")
                if name == 'parent_model':
                    object.__setattr__(self, name, value)
                else:
                    # only need when torch.__version__ >= 1.6.0
                    try:
                        remove_from(self.__dict__, self._parameters,
                                    self._buffers,
                                    self._non_persistent_buffers_set)
                    except:
                        pass
                    modules[name] = value
            elif modules is not None and name in modules:
                if value is not None:
                    raise TypeError("cannot assign '{}' as child module '{}' "
                                    "(torch.nn.Module or None expected)".format(
                                        torch.typename(value), name))
                if name == 'parent_model':
                    object.__setattr__(self, name, value)
                else:
                    modules[name] = value
            else:
                buffers = self.__dict__.get('_buffers')
                if buffers is not None and name in buffers:
                    if value is not None and not isinstance(
                            value, torch.Tensor):
                        raise TypeError(
                            "cannot assign '{}' as buffer '{}' "
                            "(torch.Tensor or None expected)".format(
                                torch.typename(value), name))
                    buffers[name] = value
                else:
                    object.__setattr__(self, name, value)


class SuperNet(nn.Module):
    def __init__(self, model, Config: OFAConfig):
        super(SuperNet, self).__init__()
        self.model = model
        self.config = Config
        self.elastic_layers = {}
        for n, m in self.model.named_modules():
            if m.__class__.__name__ in Config.SearchSpace.keys():
                Converter = ConverterMap[m.__class__.__name__]
                elastic_module = Converter(m)
                if n in self.config.static_layer:
                    elastic_module.set_static()
                replace_module(self.model, n, elastic_module)
                self.elastic_layers.update({n: elastic_module})
                del (m)

        self.init_SearchSpace()
        self.step_count = 0
        self.shrink_step = None
        self.sample_step = None
        self.stage = self.config.stages[0]
        self._update_CurrSpace(self.stage)
        self._sample_StructConfig(self.stage)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def init_SearchSpace(self):
        for n, m in self.elastic_layers.items():
            m.set_SearchSpace(deepcopy(self.config.SearchSpace[m.OriginName]))

    def set_stage(self, stage):
        self.stage = stage

    def _update_CurrSpace(self, stage):
        for n, m in self.elastic_layers.items():
            m.update_CurrSpace(stage)
            #Logger(Verbose.DEBUG)('{} CurrSpace: {}'.format(n, m.CurrSpace))

    def _sample_StructConfig(self, stage):
        sample_Structs = {}
        for n, m in self.elastic_layers.items():
            sample_StructConfig = m.sample_StructConfig(stage)
            if n in self.config.binding_layer.keys():
                sample_Structs[n] = sample_StructConfig
            #Logger(Verbose.DEBUG)('{} StructConfig: {}'.format(n, m.StructConfig))

        for main_layer in self.config.binding_layer.keys():
            for layer_name in self.config.binding_layer[main_layer]:
                sample_StructConfig = sample_Structs[main_layer]
                self.elastic_layers[layer_name].set_StructConfig(
                    sample_StructConfig)
                #Logger(Verbose.DEBUG)('{} StructConfig: {}'.format(
                #    layer_name, self.elastic_layers[layer_name].StructConfig))

    def _set_StructConfig(self, layer_name, config):
        assert layer_name in self.elastic_layers.keys(
        ), 'The `layer_name` should be a ElasticLayer'
        self.elastic_layers[layer_name].set_StructConfig(config)
        #Logger(Verbose.DEBUG)('{} StructConfig: {}'.format(
        #    layer_name, self.elastic_layers[layer_name].StructConfig))
        binding_layers = self.config.binding_layer[layer_name]
        for layer in binding_layers:
            self.elastic_layers[layer].set_StructConfig(config)
            #Logger(Verbose.DEBUG)('{} StructConfig: {}'.format(
            #    layer, self.elastic_layers[layer].StructConfig))

    def set_controller(self, shrink_step, sample_step):
        self.shrink_step = shrink_step
        self.sample_step = sample_step

    def get_SearchSpace(self):
        SearchSpace = {}
        for n, m in self.elastic_layers.items():
            SearchSpace.update({n: m.SearchSpace})
        return SearchSpace

    def get_CurrSpace(self):
        CurrSpace = {}
        for n, m in self.elastic_layers.items():
            CurrSpace.update({n: m.CurrSpace})
        return CurrSpace

    def get_StructConfig(self):
        StructConfig = {}
        for n, m in self.elastic_layers.items():
            StructConfig.update({n: m.StructConfig})
        return StructConfig

    def step(self):
        self.step_count += 1
        if self.shrink_step and self.step_count % self.shrink_step == 0:
            self._update_CurrSpace(self.stage)
        if self.sample_step and self.step_count % self.sample_step == 0:
            self._sample_StructConfig(self.stage)

    def export(self, StructConfigs=None, dummy_input=None):
        if StructConfigs is not None:
            for n, m in StructConfigs.items():
                self.elastic_layers[n].StructConfig = m
        if dummy_input is not None:
            self.model(dummy_input)
        for n, m in self.elastic_layers.items():
            static_module = m.recover()
            replace_module(self.model, n, static_module)
        return self.model
