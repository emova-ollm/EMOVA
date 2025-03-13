import os
import inspect

from transformers import AutoModelForCausalLM as _AutoModelForCausalLM, AutoConfig

from emova.registry_utils import Registry, build_from_cfg

LANGUAGE_MODEL = Registry('language_model')


def exists(x):
    return x is not None


@LANGUAGE_MODEL.register_module()
class AutoModelForCausalLM(_AutoModelForCausalLM):
    pass


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


def build_language_model(language_model_cfg, config=None, **kwargs):
    language_model_cfg.update(kwargs)
    from_pretrained = language_model_cfg.pop("from_pretrained", False) or exists(config)
    from_scratch = language_model_cfg.pop("from_scratch", False)
    trainable = language_model_cfg.pop("trainable", False)
    if from_pretrained:
        obj_cls = get_cls_from_type(language_model_cfg.pop('type'), LANGUAGE_MODEL)

        if from_scratch:
            if not exists(config):
                # config = AutoConfig.from_pretrained(language_model_cfg.pretrained_model_name_or_path)
                config = AutoConfig.from_pretrained(**language_model_cfg)
            model = obj_cls(config)
        else:
            if exists(config):
                model = obj_cls.from_pretrained(language_model_cfg.pretrained_model_name_or_path,
                                                config=config,
                                                **kwargs)
            else:
                model = obj_cls.from_pretrained(**language_model_cfg)
    else:
        model = build_from_cfg(language_model_cfg, LANGUAGE_MODEL)

    if trainable:
        model.tune()
    else:
        model.freeze()

    return model
