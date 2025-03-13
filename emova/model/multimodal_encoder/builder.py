import inspect
from abc import ABC
from torch import nn

from emova.registry_utils import Registry, build_from_cfg

MULTI_MODALITY_ENCODER = VISION_TOWER = Registry('multi_modality_encoder')


class BaseModalityEncoder(ABC, nn.Module):
    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False

    def tune(self):
        for p in self.parameters():
            p.requires_grad = True


def get_cls_from_type(obj_type, registry):
    if isinstance(obj_type, str):
        obj_cls = registry.get(obj_type)
        if obj_cls is None:
            raise KeyError(
                f'{obj_type} is not in the {registry.name} registry')
    elif inspect.isclass(obj_type) or inspect.isfunction(obj_type):
        obj_cls = obj_type
    else:
        raise TypeError(
            f'type must be a str or valid type, but got {type(obj_type)}')
    return obj_cls


def build_mm_encoder(mm_encoder_cfg, **kwargs):
    mm_encoder_cfg.update(kwargs)
    trainable = mm_encoder_cfg.pop('trainable', False)
    from_pretrained = mm_encoder_cfg.pop("from_pretrained", False)
    if from_pretrained:
        obj_cls = get_cls_from_type(mm_encoder_cfg.pop('type'), MULTI_MODALITY_ENCODER)
        model = obj_cls.from_pretrained(**mm_encoder_cfg)
    else:
        model = build_from_cfg(mm_encoder_cfg, MULTI_MODALITY_ENCODER)

    if trainable:
        model.tune()
    else:
        model.freeze()
    return model


def build_vision_tower(vision_tower_cfg, **kwargs):
    return build_mm_encoder(vision_tower_cfg, **kwargs)
