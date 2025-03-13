import os
import re

from transformers import AutoConfig

from emova.dist_utils import get_rank, synchronize


def is_zero3_model(model=None, params=None):
    if model:
        params = list(model.parameters())

    assert params
    for p in params:
        if any(hasattr(p, attr) for attr in ['ds_tensor', 'ds_id', 'ds_status', 'ds_shape', 'ds_numel']):
            return True
    else:
        return False


def load_state_dict_maybe_zero_3(model, state_dict, strict=False, ignore_status=False):
    import deepspeed
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

    params = list(model.parameters())
    msg = None
    if is_zero3_model(params=params):
        with zero.GatheredParameters(params, modifier_rank=0):
            if deepspeed.comm.get_rank() == 0:
                msg = model.load_state_dict(state_dict, strict=strict)
    else:
        msg = model.load_state_dict(state_dict, strict=strict)
    return msg
