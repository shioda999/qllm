import torch
import numpy as np
import random
from transformers import set_seed as transformers_set_seed
import math
import psutil

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    transformers_set_seed(seed)

def print_memory_usage():
    """CPUとGPUのメモリ使用量を表示"""
    # CPUメモリの使用量
    cpu_memory = psutil.virtual_memory()
    print(f"*******************\nCPU Memory Usage: {cpu_memory.used / (1024**3):.2f} GB / {cpu_memory.total / (1024**3):.2f} GB")

    # GPUメモリの使用量（CUDAが利用可能な場合）
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gpu_memory_allocated = torch.cuda.memory_allocated() / (1024**3)
        gpu_memory_reserved = torch.cuda.memory_reserved() / (1024**3)
        print(f"GPU Memory Allocated: {gpu_memory_allocated:.2f} GB")
        print(f"GPU Memory Reserved: {gpu_memory_reserved:.2f} GB")
    else:
        print("CUDA is not available.")
    print('*******************')

def isinstance_str(x: object, cls_name: str):
    """
    Checks whether x has any class *named* cls_name in its ancestry.
    Doesn't require access to the class's implementation.
    
    Useful for patching!
    """

    for _cls in x.__class__.__mro__:
        if _cls.__name__ == cls_name:
            return True
    
    return False

@torch.no_grad()
def calc_init_scale(x, Qp, per_channel=False, method="l2-norm"):
    if per_channel: x = x.reshape(x.shape[0], -1)
    if method == "minmax":
        if per_channel:
            #return (torch.max(x, dim=1))/(math.sqrt(Qp))
            return (torch.max(x, dim=1))/Qp
        else:
            #return torch.max(torch.abs(x))/(math.sqrt(Qp))
            return torch.max(torch.abs(x))/Qp
    elif method == "percentile":
        p = 0.000001
        if per_channel:
            low = np.quantile(x.cpu(), p, dim=1)
            high = np.quantile(x.cpu(), 1-p, dim=1)
        else:
            low = np.quantile(x.cpu(), p)
            high = np.quantile(x.cpu(), 1-p)
        return torch.tensor(np.maximum(-low, high) / Qp, device=x.device)
        # low = torch.quantile(x.cpu(), p, dim=0).max()
        # high = torch.quantile(x.cpu(), 1-p, dim=0).max()
        # return torch.maximum(-low, high) / Qp * 1.25
    elif method == "ternary":
        if per_channel:
            return torch.mean(torch.abs(x), dim=1) / Qp
        else:
            return torch.mean(torch.abs(x)) / Qp * 2.5
    elif method == "ternary_percentile":
        if per_channel:
            return torch.mean(torch.abs(x), dim=1) / Qp
        else:
            p = 0.01
            x = x.abs()
            th = torch.quantile(x, 1-p, dim=0).max()
            x = x[x < th]
            return torch.mean(x) / Qp * 3
    elif method == "absmean":
        if per_channel:
            return torch.mean(torch.abs(x), dim=1)*2/(math.sqrt(Qp))
        else:
            return torch.mean(torch.abs(x))*2/(math.sqrt(Qp))
    elif method == "l2-norm":
        if per_channel:
            return torch.mean(torch.abs(x).pow(2), dim=1).sqrt()*2 / Qp
        else:
            return torch.mean(torch.abs(x).pow(2)).sqrt()*2 / Qp
    elif method == "mean-std":
        if per_channel:
            mean = torch.mean(x, dim=1)
            std = torch.std(x, dim=1)
            return torch.max(torch.stack([torch.abs(mean-3*std), torch.abs(mean + 3*std)]), dim=0)[0] / Qp * 1.25
        else:
            mean = x.mean()
            std = x.std()
            if Qp == 1: return max([torch.abs(mean-3*std), torch.abs(mean+3*std)]) / Qp * 0.5
            else: return max([torch.abs(mean-3*std), torch.abs(mean+3*std)]) / Qp * 1.25
    else:
        raise ValueError()