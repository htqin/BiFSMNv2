class Registry(object):
    def __init__(self, name):
        self._name = name
        self._module_dict = dict()

    @property
    def name(self):
        return self._name

    @property
    def module_dict(self):
        return self._module_dict

    def module_names(self):
        return list(self._module_dict.keys())

    def register(self, cls):
        module_name = cls.__name__
        if module_name in self._module_dict:
            raise KeyError('{} is already registered in {}'.format(
                module_name, self.name))
        self._module_dict[module_name] = cls
        return cls

    def get(self, name):
        if name in self._module_dict:
            return self._module_dict[name]
        else:
            raise KeyError('{} is not registered in {}'.format(name, self.name))


CONFIG = Registry('config')
