#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pickle
import torch
import torch.distributed as dist

__all__ = [
    'get_rank', 'get_world_size', 'is_main_process', 'ddp_barrier',
    'synchronize', 'all_reduce', 'all_gather_tensor', 'all_gather'
]


def get_world_size():
    """
    get distributed worker number
    :return:
    """
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def ddp_barrier():
    """In DDP mode, this function will synchronize all processes.
    torch.distributed.barrier() will block processes until the whole
    group enters this function.
    """
    if torch.distributed.is_initialized():
        torch.distributed.barrier()


def get_rank():
    """
    get current rank
    :return: rank id if enable distributed
    """
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def synchronize():
    """
       Helper function to synchronize (barrier) among all processes when
       using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()


def is_main_process():
    """
    check current process is main or not
    :return: return True if current process is the main else return False
    """
    return get_rank() == 0


def all_reduce(x, reduction='sum'):
    """
    do reduction(sum) across multi-gpus
    :param x: tensor or list of tensors:
    :param reduction: reduction type, support "sum" and "mean"
    :return: reduced tensor
    """
    if get_world_size() <= 1:
        return x
    assert reduction in [
        'sum', 'mean'
    ], 'only support reduction type: "sum", "mean", got {}'.format(reduction)
    is_dict = isinstance(x, dict)
    if is_dict:
        for k, v in x.items():
            rt = v.clone()
            dist.all_reduce(rt, op=torch.distributed.ReduceOp.SUM)
            if reduction == 'mean':
                rt /= get_world_size()
            x[k] = rt
        return x
    else:
        rt = x.clone()
        dist.all_reduce(rt, op=torch.distributed.ReduceOp.SUM)
        if reduction == 'mean':
            rt /= get_world_size()
        return rt


def all_gather_tensor(tensor):
    """
    Run all_gather on tensors, all tensors must with same dtype
    Args:
        tensor: tensor on each rank, can be different in size and shape
    Returns:
        list tensor: list of tensors from each rank
    """
    with torch.no_grad():
        world_size = get_world_size()
        rank = get_rank()
        if world_size == 1:
            return [tensor]

        # transfer tensor to gpu
        if not tensor.is_cuda:
            tensor = tensor.cuda()

        # gathering tensors of different shapes
        tensor_list = []

        # obtain Tensor size and dims of each rank
        local_size = int(tensor.numel())
        local_dim = int(tensor.dim())
        local_size_dims = torch.as_tensor([local_size, local_dim],
                                          dtype=torch.int64).cuda()

        # all gather the max size and max dims from all ranks
        size_dims_list = [
            torch.as_tensor([0, 0], dtype=torch.int64).cuda()
            for _ in range(world_size)
        ]
        dist.all_gather(size_dims_list, local_size_dims)
        size_list = [int(size[0].item()) for size in size_dims_list]
        dims_list = [int(dim[1].item()) for dim in size_dims_list]

        max_dims = max(dims_list)
        max_size = max(size_list)

        # obtain original shape, dtype
        tensor_shape = [i for i in tensor.shape]
        if len(tensor_shape) < max_dims:
            tensor_shape.extend(
                [0 for _ in range(max_dims - len(tensor_shape))])
        tensor_shape = torch.as_tensor([tensor_shape], dtype=torch.int64).cuda()

        # all gather the shape from all ranks
        shape_list = [
            torch.zeros(size=(max_dims, ), dtype=torch.int64).cuda()
            for _ in range(world_size)
        ]
        dist.all_gather(shape_list, tensor_shape)

        # receiving Tensor from all ranks
        # we pad the tensor because torch all_gather does not support
        for _ in range(world_size):
            tensor_list.append(
                torch.zeros(size=(max_size, ), dtype=tensor.dtype).cuda())

        if local_size != max_size:
            padding = torch.zeros(size=(max_size - local_size, ),
                                  dtype=tensor.dtype).cuda()
            tensor = torch.cat((tensor.view(-1), padding), dim=0)

        dist.all_gather(tensor_list, tensor)

        res = list()
        for i in range(world_size):
            shape = shape_list[i].cpu().tolist()[:dims_list[i]]
            tmp = tensor_list[i][:size_list[i]].view(shape)
            res.append(tmp)
        return res


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).cuda()

    # obtain Tensor size of each rank
    local_size = torch.LongTensor([tensor.numel()]).cuda()
    size_list = [torch.LongTensor([0]).cuda() for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.ByteTensor(size=(max_size, )).cuda())
    if local_size != max_size:
        padding = torch.ByteTensor(size=(max_size - local_size, )).cuda()
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list
