"""
This file contains primitives for multi-gpu communication.
This is useful when doing distributed training.
"""
import os
import pickle

import torch
import torch.distributed as dist


def is_distributed():
    return get_world_size() > 1


def get_world_size():  # default data-parallel world-size
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_rank():  # default data-parallel rank
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_local_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    local_rank = int(os.getenv('LOCAL_RANK', 0))
    return local_rank


def is_master():
    return get_rank() == 0


def is_local_master():
    return get_local_rank() == 0


def get_local_proc_group(group_size=8):
    if dist.get_world_size() <= group_size:
        return None
    if not hasattr(get_local_proc_group, 'process_groups'):
        num_groups = dist.get_world_size() // group_size  # 8 processes per node by default
        groups = [list(range(i * group_size, (i + 1) * group_size)) for i in range(num_groups)]
        get_local_proc_group.process_groups = [torch.distributed.new_group(group) for group in groups]
    group_idx = get_rank() // group_size
    process_groups = get_local_proc_group.process_groups[group_idx]
    return process_groups


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


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    to_device = torch.device("cuda")
    # to_device = torch.device("cpu")

    world_size = get_world_size()
    if world_size == 1:
        return [data]

    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to(to_device)

    # obtain Tensor size of each rank
    local_size = torch.LongTensor([tensor.numel()]).to(to_device)
    size_list = [torch.LongTensor([0]).to(to_device) for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.ByteTensor(size=(max_size,)).to(to_device))
    if local_size != max_size:
        padding = torch.ByteTensor(size=(max_size - local_size,)).to(to_device)
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.reduce(values, dst=0)
        if dist.get_rank() == 0 and average:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


def broadcast(data, src=0, group=None):
    """
    Run broadcast on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
        decvie: current device
    Returns:
        data: list of data gathered from each rank
    """
    to_device = torch.device("cuda")

    world_size = get_world_size()
    if world_size == 1:
        return data

    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to(to_device)

    # obtain Tensor size of each rank
    local_size = torch.LongTensor([tensor.numel()]).to(to_device)
    size_list = [torch.LongTensor([0]).to(to_device) for _ in range(world_size)]
    dist.all_gather(size_list, local_size, group=group)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    if local_size != max_size:
        padding = torch.ByteTensor(size=(max_size - local_size,)).to(to_device)
        tensor = torch.cat((tensor, padding), dim=0)

    dist.broadcast(tensor, src=src, group=group, async_op=False)
    buffer = tensor.cpu().numpy().tobytes()
    return pickle.loads(buffer)


def gather_compute(feat1, feat2):
    world_size = get_world_size()

    if world_size > 1:
        output = [torch.zeros_like(feat2) for _ in range(dist.get_world_size())]
        dist.all_gather(output, feat2)
        feat2_gather = torch.cat(output)
    else:
        feat2_gather = feat2.detach()
    return feat1 @ feat2_gather.T
