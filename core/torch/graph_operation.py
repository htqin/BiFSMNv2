import torch
import torch.nn as nn


def replace_module(model: nn.Module, name: str, module: nn.Module):
    scopes = name.split('.')
    current_scope = ''
    current_module = model
    for scope in scopes[:-1]:
        current_scope += scope
        assert hasattr(
            current_module,
            scope), '{} scope is not in the model'.format(current_scope)
        current_module = getattr(current_module, scope)
    current_scope += scopes[-1]
    scope = scopes[-1]
    assert hasattr(current_module,
                   scope), '{} scope is not in the model'.format(current_scope)
    current_module.add_module(scope, module)


def find_module_by_name(model: nn.Module, name: str):
    for n, m in model.named_modules():
        if n == name:
            return m
