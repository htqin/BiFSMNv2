from abc import abstractmethod
from copy import deepcopy
from .torch.ops import OpBase, OPS
from .logger import Logger, Verbose, set_slim_logger_level

set_slim_logger_level(Verbose.INFO)


class ConfigBase(object):
    """
    a base class for slim configuration
    """
    def __init__(self, config, framework):
        super(ConfigBase, self).__init__()
        self.__key = list()
        self.skip_scope = []
        self.work_scope = []
        self.strict_scope = False
        self.type_config = {}
        self.layer_config = {}
        assert framework in [
            'torch', 'tf'
        ], "The `framework` should only be one of ['torch', 'tf']"
        self.framework = framework
        self.parse_valid(config)

    def set_type_config(self, op_type, config_op: OpBase):
        self.type_config[op_type] = deepcopy(config_op)

    def set_layer_config(self, layer_name, config_op: OpBase):
        self.layer_config[layer_name] = deepcopy(config_op)

    def get_type_config(self, op_type) -> OpBase:
        return deepcopy(self.type_config[op_type])

    def get_layer_config(self, layer_name) -> OpBase:
        return deepcopy(self.layer_config[layer_name])

    def __getitem__(self, item):
        if item in self.__key:
            return getattr(self, item)

    @abstractmethod
    def parse_valid(self, params: dict):
        raise NotImplementedError('parse_valid is not implemented for ',
                                  self.__class__.__name__)

    def NeedSkip(self, op_name, op_type, framework):
        need_skip = False
        if 'gradients' in op_name and framework == 'tf':
            return True
        if op_type not in self.type_config.keys():
            need_skip = True

        if len(self.work_scope) > 0:
            if self.strict_scope:
                if op_name not in self.work_scope:
                    need_skip = True
            else:
                hit = False
                for scope in self.work_scope:
                    if op_name.startswith(scope):
                        hit = True
                        break
                if not hit:
                    need_skip = True

        if self.strict_scope:
            if op_name in self.skip_scope:
                Logger(Verbose.INFO)('Strict Skip: {}'.format(op_name))
                need_skip = True
        for scope in self.skip_scope:
            if op_name.startswith(scope):
                need_skip = True
                Logger(Verbose.INFO)('Skip: {}'.format(op_name))
                break

        return need_skip


class OpConfig(object):
    def __init__(self):
        raise NotImplementedError('It is not impolemented for ',
                                  self.__class__.__name__)

    def fill_with_default(self, OpConfig: dict, default_config: dict):
        for key in default_config.keys():
            if key not in OpConfig:
                OpConfig[key] = default_config[key]
        for k, v in OpConfig.items():
            self.__dict__[k] = v

    def parse_for_torch(self):
        pass

    def parse_for_tf(self):
        pass

    def __str__(self):
        return self.__dict__
