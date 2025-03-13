import importlib.util
import json
import os
from copy import deepcopy
from typing import Any


class DotDict(dict):
    """A dictionary that supports dot notation for accessing its keys."""

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"No such attribute: {item}")

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, item):
        del self[item]

    @staticmethod
    def convert(d):
        """Recursively convert a dictionary and its nested dictionaries into DotDicts."""
        if isinstance(d, dict):
            return DotDict({k: DotDict.convert(v) for k, v in d.items()})
        elif isinstance(d, list):
            return [DotDict.convert(v) for v in d]
        return d


def load_py_config(config_path):
    config_path = os.path.abspath(config_path)
    spec = importlib.util.spec_from_file_location("config", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    cfg_dict = {key: value for key, value in config.__dict__.items() if not key.startswith('_')}

    if '_base_' in config.__dict__:
        cfg_dict['_base_'] = config.__dict__['_base_']

    return cfg_dict


def merge_dicts(base, new):
    base = deepcopy(base)
    for key, value in new.items():
        if isinstance(value, dict) and key in base and isinstance(base[key], dict):
            base[key] = merge_dicts(base[key], value)
        else:
            base[key] = value
    return base


def load_and_merge_config(config_path):
    config_path = os.path.abspath(config_path)
    cfg = load_py_config(config_path)

    if '_base_' in cfg:
        base_cfg = {}
        base_files = cfg.pop('_base_')

        if not isinstance(base_files, list):
            base_files = [base_files]

        for base_file in base_files:
            base_file_path = os.path.join(os.path.dirname(config_path), base_file)
            base_cfg = merge_dicts(base_cfg, load_and_merge_config(base_file_path))

        cfg = merge_dicts(base_cfg, cfg)

    return cfg


class Config:
    def __init__(self, cfg_dict=None):
        # 将 cfg_dict 递归转换成 DotDict
        self._cfg_dict = DotDict.convert(cfg_dict) if cfg_dict else DotDict()

    @classmethod
    def fromfile(cls, filename):
        cfg = load_and_merge_config(filename)
        return cls(cfg)

    def __getitem__(self, key):
        return self._cfg_dict[key]

    def __getattr__(self, name):
        try:
            return self._cfg_dict[name]
        except KeyError:
            raise AttributeError(f"'Config' object has no attribute '{name}'")

    def __repr__(self):
        return f"Config({self._cfg_dict})"

    def __len__(self):
        return len(self._cfg_dict)

    def set_nested(self, key_path: str, value: Any):
        """
        Set a value in a nested dictionary or dataclass based on a dot-separated key path.
        """
        keys = key_path.split('.')
        obj = self
        for key in keys[:-1]:
            if isinstance(obj, dict):
                obj = obj.setdefault(key, {})
            else:
                obj = getattr(obj, key, None)
                if obj is None:
                    raise AttributeError(f"Attribute '{key}' not found in the config.")
        last_key = keys[-1]

        # Attempt to convert the value to an appropriate type
        try:
            if isinstance(value, str):
                if value.lower() in ['true', 'false']:
                    value = value.lower() == 'true'
                else:
                    value = eval(value)  # Use with caution
        except:
            pass  # Keep as string if conversion fails

        if isinstance(obj, dict):
            obj[last_key] = value
        else:
            setattr(obj, last_key, value)

    def pretty(self):
        return json.dumps(self._cfg_dict, indent=4)

    def items(self):
        return self._cfg_dict.items()


class Registry:
    def __init__(self, name):
        self._name = name
        self._module_dict = {}

    @property
    def module_names(self):
        return list(self._module_dict.keys())

    def register_module(self, module=None, name=None):
        if module is None:
            def _register(cls):
                module_name = name if name else cls.__name__
                if module_name in self._module_dict:
                    raise KeyError(f'{module_name} is already registered in {self._name}')
                self._module_dict[module_name] = cls
                return cls

            return _register
        else:
            module_name = name if name else module.__name__
            if module_name in self._module_dict:
                raise KeyError(f'{module_name} is already registered in {self._name}')
            self._module_dict[module_name] = module
            return module

    def get(self, name):
        return self._module_dict.get(name)


def build_from_cfg(cfg, registry, default_args=None):
    if not isinstance(cfg, dict):
        raise TypeError('cfg must be a dict')
    if 'type' not in cfg:
        raise KeyError('cfg must contain the key "type"')

    args = deepcopy(cfg)
    obj_type = args.pop('type')

    if default_args is not None:
        for name, value in default_args.items():
            args.setdefault(name, value)

    if isinstance(obj_type, str):
        obj_cls = registry.get(obj_type)
        if obj_cls is None:
            raise KeyError(f'{obj_type} is not in the {registry._name} registry')
    else:
        raise TypeError(f'type must be a str, but got {type(obj_type)}')

    return obj_cls(**args)


if __name__ == '__main__':
    LANGUAGE_MODEL = Registry('language_model')

    @LANGUAGE_MODEL.register_module()
    class MyLanguageModel:
        def __init__(self, param1, param2):
            self.param1 = param1
            self.param2 = param2


    cfg = {
        'type': 'MyLanguageModel',
        'param1': 'value1',
        'param2': 'value2'
    }

    model = build_from_cfg(cfg, LANGUAGE_MODEL)
    print(model.param1)
    print(model.param2)
