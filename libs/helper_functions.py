import time
import platform
from pynvml import *
import torch
import yaml


def get_configs(path: str):
    """return defined configs in yaml file"""
    params = yaml.safe_load(open(path, "r", encoding="utf-8"))
    return params


def get_executing_time(start_time):
    """
    start time as param passing with time.time()
    based on this start time to calculate the executing time of
    a function
    :param start_time
    :return executing time
    """
    end_time = time.time()  # calling current time
    result = end_time - start_time  # executing time
    return result


# setup device
def get_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return device


# GPU utilization
def get_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB")


# get GPU services
def get_gpu_services():
    """
    The following code is used for checking CUDA service
    quantities of CUDA services prepared for training.
    Using torch.cuda
    """
    is_cuda_free = torch.cuda.is_available()
    print(f"CUDA free: {is_cuda_free}")  # checking cuda service
    # get number of CUDA services
    num_cudas = torch.cuda.device_count()
    print(f"Number of devices: {num_cudas}")
    if is_cuda_free:
        for i in range(num_cudas):
            # get device props
            device = torch.device("cuda", i)
            print(
                f"CUDA device: {i} Name: {torch.cuda.get_device_name(i)}"
                f"Compute capability: {torch.cuda.get_device_capability(i)}"
                f"Total memory: {torch.cuda.get_device_properties(i).total_memory} bytes"
                f"Props: {torch.cuda.get_device_properties(i)}"
            )
    # get CPU
    print(
        f"CPU: {platform.processor()}"
        f"System: {platform.system(), platform.release()}"
        f"Py Version: {platform.python_version()}"
    )
