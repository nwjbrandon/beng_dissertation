import importlib

import yaml


def load_config(file_path):
    with open(file_path) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        return cfg


def import_module(module_path):
    module_name, _, class_name = module_path.rpartition(".")
    m = importlib.import_module(module_name)

    return getattr(m, class_name)
