import struct
import torch
import numpy as np
import rocksdb_binding as rocksdb
from typing import Tuple, Dict, Optional, List
import os
from concurrent.futures import ThreadPoolExecutor, Future, TimeoutError
from typing import Optional
import threading

import threading

class KVStorage:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(KVStorage, cls).__new__(cls)
                cls._instance._first_init_args = (args, kwargs)
            else:
                # Check if parameters match the original
                if (args, kwargs) != cls._instance._first_init_args:
                    raise ValueError(
                        f"KVStorage singleton already created with different parameters: "
                        f"expected {cls._instance._first_init_args}, got {(args, kwargs)}"
                    )
            return cls._instance

    def __init__(
        self,
        dtype: torch.dtype,
        head_num: int,
        head_dim: int,
        layer_num: int,
        executor_worker_num: int = 4,
        db_path: str = "~/kv4kv/KVS",
    ):
        if getattr(self, '_initialized', False):
            return
        self._initialized = True

        self.dtype = dtype
        self.head_num = head_num
        self.head_dim = head_dim
        self.layer_num = layer_num
        self.db_path = os.path.expanduser(db_path)

        self.db = rocksdb.RocksDB()
        print(f"Opening RocksDB at '{self.db_path}'")
        open_status = self.db.open(self.db_path)
        assert open_status

        self.executor = ThreadPoolExecutor(max_workers=executor_worker_num)

    def __del__(self):
        # is it necessary?
        pass
        # if hasattr(self, 'db'):
        #     del self.db
        #     self.db = None
        #     print(f"RocksDB instance at '{self.db_path}' closed.")
        
        # if hasattr(self, 'executor'):
        #     self.executor.shutdown(wait=True)

    def _make_key(self, key: torch.Tensor) -> bytes:
        """Prefix bytes → RocksDB key"""
        if isinstance(key, torch.Tensor):
            return key.cpu().numpy().tobytes()
        elif isinstance(key, bytes):
            return key
        elif isinstance(key, list):
            assert isinstance(key[0], int), "List keys must contain integers"
            return np.array(key, dtype=np.int32).tobytes()

    def put_prefix_kv(
        self, 
        key: torch.Tensor, 
        # shape: [2, layer_num, pre_len, head_num, head_dim]
        kv_tensor: torch.Tensor
    ):
        _2, layer_num, pre_len, head_num, head_dim = kv_tensor.shape
        if layer_num != self.layer_num or head_num != self.head_num or \
            head_dim != self.head_dim or pre_len != key.shape[0]:
            raise ValueError("invalid shape of kv_tensor")
        
        for L in range(1, pre_len + 1):
            prefix_ids = key[:L]  # Prefix of length L
            prefix_tensor = kv_tensor[:, :, :L, :, :]
            db_key = self._make_key(prefix_ids)
            value = prefix_tensor.to(torch.float32).cpu().numpy().tobytes()
            put_status = self.db.put(db_key, value)
            assert put_status

    def probe_max_prefix(
        self, 
        key: torch.Tensor, 
        min_length: int,
        max_length: int
    ) -> Tuple[int, Optional[Future]]:
        matched_pre_len = min_length
        for pre_len in range(min_length + 1, max_length + 1):
            db_key = self._make_key(key[:pre_len])
            result = self.db.get(db_key)
            if result is None:
                break
            else:
                matched_pre_len = pre_len
        
        if matched_pre_len != 0:
            matched_key = key[:matched_pre_len]
            
            # issue a worker thread to perform _rocksdb_get
            kv_future: Future = self.executor.submit(
                self._rocksdb_get,
                matched_key,
                torch.device("cuda")
            )
                
            return matched_pre_len, kv_future
        else:
            return matched_pre_len, None

    def _rocksdb_get(
        self,
        matched_key: torch.Tensor | List[int],
        device: torch.device = torch.device("cuda"),
    ) -> torch.Tensor:
        # disk to cpu          
        kv_cpu_raw = self.db.get(self._make_key(matched_key))   
        # cpu reshape
        kv_np = np.frombuffer(kv_cpu_raw, dtype=np.float32).reshape(
            2, self.layer_num, matched_key.shape[0], self.head_num, self.head_dim
        )
        kv_tensor = torch.from_numpy(kv_np.copy())
        # cpu to gpu
        kv_tensor = kv_tensor.to(device=device, dtype=self.dtype)
                
        return kv_tensor

    def wait_for_kv(
        self,
        kv_future: Future,
        timeout: Optional[float] = None,  # seconds; None = wait forever
    ) -> torch.Tensor:
        kv_tensor = kv_future.result(timeout=timeout)
        _2, layer_num, pre_len, head_num, head_dim = kv_tensor.shape
        if layer_num != self.layer_num or head_num != self.head_num or \
            head_dim != self.head_dim:
            raise ValueError("invalid shape of kv_tensor")
        return kv_tensor
        
        
        


if __name__ == "__main__":
    head_num = 2
    head_dim = 4
    layer_num = 8
    kvs = KVStorage(
        dtype=torch.float16,
        head_num=head_num,
        head_dim=head_dim,
        layer_num=layer_num,
        executor_worker_num=4,
        db_path="~/kv4kv/test",
    )
    
    key1 = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.int32)
    kv_tensor1 = torch.randn(2, layer_num, 6, head_num, head_dim, dtype=torch.float16)
    kvs.put_prefix_kv(key1, kv_tensor1)
    
    
    matched_pre_len, kv_future = kvs.probe_max_prefix(
        key=torch.tensor([1, 2, 3, 4, 5, 6, 7], dtype=torch.int32),
        min_length=0,
        max_length=7,
    )
    kv = kv_future.result() if kv_future else None
    assert matched_pre_len == 6 
    assert torch.equal(kv.cpu(), kv_tensor1)
    
    matched_pre_len, kv_future = kvs.probe_max_prefix(
        key=torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.int32),
        min_length=0,
        max_length=6,
    )
    kv = kv_future.result() if kv_future else None
    assert matched_pre_len == 6 
    assert torch.equal(kv.cpu(), kv_tensor1)
    
    matched_pre_len, kv_future = kvs.probe_max_prefix(
        key=torch.tensor([1, 2, 3, 4, 5], dtype=torch.int32),
        min_length=0,
        max_length=5,
    )
    kv = kv_future.result() if kv_future else None
    assert matched_pre_len == 5 
    assert torch.equal(kv.cpu(), kv_tensor1[:, :, :5, :, :])