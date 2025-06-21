import abc
import logging
import threading
import torch
from enum import IntEnum
from functools import wraps
from typing import List, Optional, Tuple, Union

class KVStorage:
    def __init__(
        self,
        dtype: torch.dtype,
        head_num: int,
        head_dim: int,
        layer_num: int,
    ):
        # initialize metadata of KV Storage
        pass
    
    def get_prefix(
        input_ids: torch.Tensor,    
        layer_id: int
    ):
        prefix = None
        prefix_len = None
        kv_loc = None
 
        return prefix, prefix_len, kv_loc
    
    def get_kv_cache():
        kv_tensor = None
        return kv_tensor

    def put_prefix(
        input_ids: torch.Tensor,
        # shape: (2, head_num, head_dim)
        kv_tensor: torch.Tensor,
        layer_id: int
    ):
        pass
