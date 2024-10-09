from typing import Dict, Callable, List
import torch
import torch.nn as nn
import numpy as np
import random
import os

def set_seed_everywhere(seed):
    '''
    Set seed for reproducibility.
    '''
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def enable_deterministic_run():
    '''
    Enable deterministic run.
    '''
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def get_device(device: str = "cuda"):
    '''
    Get device for training.
    '''
    if  device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")
    
def replace_submodules(
        root_module: torch.nn.Module,
        predicate: Callable[[torch.nn.Module], bool],
        func: Callable[[torch.nn.Module], torch.nn.Module]) -> torch.nn.Module:
    """
    predicate: Return true if the module is to be replaced.
    func: Return new module to use.
    """
    if predicate(root_module):
        return func(root_module)
    bn_list = [k.split(".") for k, m 
               in root_module.named_modules() 
               if predicate(m)]
    for *parent, k in bn_list:
        # get parent module
        parent_module = root_module
        if len(parent) > 0:
            # join parent names by '.' to get submodule name
            parent_module = root_module.get_submodule('.'.join(parent))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # verify that all BN are replaced
    bn_list = [k.split('.') for k, m 
        in root_module.named_modules(remove_duplicate=True) 
        if predicate(m)]
    assert len(bn_list) == 0
    return root_module

def dict_apply(input:dict,
               fn:Callable[[torch.Tensor], torch.Tensor]) -> dict:
    '''
    Apply function to all tensors in dictionary recursively.
    '''
    result = dict()
    for k, v in input.items():
        if isinstance(v, dict):
            result[k] = dict_apply(v, fn)
        elif isinstance(v, torch.Tensor):
            result[k] = fn(v)
    return result


def get_leading_dims(batch_tensor: torch.Tensor, single_tensor_shape) -> torch.Size:
    '''
    Get leading dimensions of tensor.
    '''
    if isinstance(single_tensor_shape, int):
        single_tensor_shape = (single_tensor_shape,)
    leading_dim_num = batch_tensor.dim() - len(single_tensor_shape)
    return batch_tensor.size()[:leading_dim_num]
    

