import re
import torch
from collections import OrderedDict
from byteslim.core.logger import Logger, Verbose

OmitType = [
    'prim::Constant', 'prim::ListConstruct', 'prim::GetAttr', 'aten::to',
    'aten::Int', 'aten::size'
]


class BindSet(object):
    def __init__(self):
        super(BindSet, self).__init__()
        self.cur_set = 0
        self.ObjectSet = {}
        self.SetObject = {}

    def binding(self, a, b):
        if a not in self.ObjectSet and b not in self.ObjectSet:
            self.ObjectSet.update({a: self.cur_set, b: self.cur_set})
            self.SetObject.update({self.cur_set: [a]})
            self.SetObject[self.cur_set].append(b)
            self.cur_set += 1
        elif a in self.ObjectSet and b in self.ObjectSet:
            pass
        elif a in self.ObjectSet:
            self.ObjectSet[b] = self.ObjectSet[a]
            self.SetObject[self.ObjectSet[a]].append(b)
        else:
            self.ObjectSet[a] = self.ObjectSet[b]
            self.SetObject[self.ObjectSet[b]].append(a)

    def __getattr__(self, obj):
        if obj not in self.ObjectSet:
            return []
        else:
            return self.SetObject[self.ObjectSet[obj]]


class SlimOp(object):
    def __init__(self, node, graph):
        super(SlimOp, self).__init__()
        self.nodes = [node]
        self.graph = graph
        self.types = [node.kind() for node in self.nodes]
        self.scope = self.nodes[0].scopeName().split('/')[-1].replace(
            '__module.', '')
        self.is_leaf = self._is_leaf()
        self.ModuleName = self._GetName()
        self.ModuleClass = self._GetModuleClass()
        self.prune_config = {}

    def _is_leaf(self):
        name = self.scope
        for n, m in self.graph.trace.named_modules():
            if n == name:
                is_leaf = len(list(m.named_children())) == 0
                return is_leaf
        raise BaseException("Can not find {} module in model".format(name))

    def _GetName(self):
        name = self.scope
        if not self.is_leaf:
            name = name + '.' + self._AtenToModule(self.types[0])
            if name + '.' + 'slim_func' in self.graph._ops.keys():
                i = 0
                while name + '.' + str(i) in self.graph._ops.keys():
                    i += 1
                name = name + '.' + str(i)
            name = name + '.' + 'slim_func'
        return name

    def inputs(self):
        all_inputs = []
        all_outputs = []
        op_inputs = []
        for node in self.nodes:
            all_inputs.extend(list(node.inputs()))
            all_outputs.extend(list(node.outputs()))
        for Input in all_inputs:
            if Input not in all_outputs and Input.node().kind() not in OmitType:
                op_inputs.append(Input)
        return op_inputs

    def outputs(self):
        all_inputs = []
        all_outputs = []
        op_outputs = []
        for node in self.nodes:
            all_inputs.extend(list(node.inputs()))
            all_outputs.extend(list(node.outputs()))
        for Output in all_outputs:
            if Output not in all_inputs and Output.node().kind(
            ) not in OmitType:
                op_outputs.append(Output)
        return op_outputs

    def _AtenToModule(self, AtenName):
        if not AtenName.startswith('aten::'):
            return AtenName
        AtenName = AtenName[6:]
        ModuleName = ''
        toUpper = True
        for c in AtenName:
            if c == '_':
                toUpper = True
                continue
            if toUpper:
                c = c.upper()
                toUpper = False
            ModuleName += c
        return ModuleName

    def _GetModuleClass(self):
        if self.is_leaf:
            for n, m in self.graph.trace.named_modules():
                if self.ModuleName == n:
                    return m._name
        else:
            assert len(
                self.types
            ) == 1, 'Convert a function in graph to a Module should to assure that op only have one node'
            return self._AtenToModule(self.types[0])

    def insert_node(self, node):
        for i, op_node in enumerate(self.nodes):
            for output in node.outputs():
                if output in op_node.inputs():
                    self.nodes.insert(i, node)
                    self.types.insert(i, node.kind())
                    return
        self.nodes.append(node)
        self.types.append(node.kind())

    def __repr__(self):
        next_ops = [op.ModuleName for op in self.graph.next_ops(self)]
        string = '\nName  : ' + self.ModuleName + '   '
        string += 'Class  :' + self.ModuleClass + '\n'
        string += 'output: ' + str(next_ops) + '\n'
        return string


class SlimGraph(object):
    def __init__(self, model, dummy_input):
        super(SlimGraph, self).__init__()
        self.trace = torch.jit.trace(model, dummy_input)
        torch._C._jit_pass_inline(self.trace.graph)
        self.graph = self.trace.graph

        self.binding_layers = BindSet()
        self.RelationOps = ['Add']
        self.DominatingOps = ['Conv2d', 'Conv1d', 'ConvTranspose1d', 'Linear']

    def get_ops(self):
        if not hasattr(self, '_ops'):
            self._ops = OrderedDict()
            nodes = [
                node for node in self.graph.nodes() if self._is_OpNode(node)
            ]
            for node in nodes:
                fake_op = SlimOp(node, self)
                if fake_op.ModuleName in self._ops.keys():
                    self._ops[fake_op.ModuleName].insert_node(node)
                else:
                    self._ops.update({fake_op.ModuleName: fake_op})
        return list(self._ops.values())

    def _is_OpNode(self, node):
        is_op = True
        if node.kind() in OmitType:
            is_op = False
        elif not node.kind().startswith('aten::'):
            is_op = False
        return is_op

    def get_op_by_type(self, type):
        return [op for op in self.get_ops() if op.type == type]

    def get_op_by_ModuleName(self, ModuleName):
        return [op for op in self.get_ops() if op.ModuleName == ModuleName]

    def next_ops(self, op: SlimOp):
        outputs = op.outputs()
        next_ops = []
        for op in self.get_ops():
            for output in outputs:
                if output in op.inputs():
                    next_ops.append(op)
        return next_ops

    def pre_ops(self, op: SlimOp):
        inputs = op.inputs()
        pre_ops = []
        for op in self.get_ops():
            for input in inputs:
                if input in op.outputs():
                    pre_ops.append(op)
        return pre_ops

    def get_tensor_shape(self, tensor):
        out_info = re.search('%.*? : .*?\(.*?\)', str(tensor)).group()
        shape_info = re.search('\(.*\)', out_info).group()[1:-1]
        shape = [int(i) for i in shape_info.split(', ')]
        return shape
