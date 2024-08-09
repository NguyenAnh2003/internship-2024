import time
import platform
import torch
import yaml


def get_configs(path: str):
    """return defined configs in yaml file"""
    params = yaml.safe_load(open(path, "r", encoding="utf-8"))
    return params

