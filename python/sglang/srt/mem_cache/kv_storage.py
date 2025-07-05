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
        db_path: str = "db",
    ):
        import inspect
        for i in inspect.stack():
            print(f"{i.filename}:{i.lineno} - Function: {i.function}")
        print(f"[KVStorage::__init__] Initializing KVStorage with dtype={dtype}, kvtensor shape=({2}, {layer_num}, seq_len, {head_num}, {head_dim})")

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

    def _make_key(self, key: List[int]) -> bytes:
        """Prefix bytes → RocksDB key"""
        if isinstance(key, list):
            assert isinstance(key[0], int), "List keys must contain integers"
            return np.array(key, dtype=np.int32).tobytes()
        else:
            raise TypeError("Key must be a list of integers")

    def put_prefix_kv(
        self, 
        key: torch.Tensor, 
        # shape: [2, layer_num, pre_len, head_num, head_dim]
        kv_tensor: torch.Tensor
    ):
        required_shape = (2, self.layer_num, len(key), self.head_num, self.head_dim)
        if kv_tensor.shape != required_shape:
            # print shapes
            print(f"[KVStorage::put_prefix_kv] {required_shape=}")
            print(f"[KVStorage::put_prefix_kv] {kv_tensor.shape=}")
            raise ValueError("invalid shape of kv_tensor")
        print(f"[KVStorage::put_prefix_kv] Storing {kv_tensor.shape=}")
        for L in range(1, len(key) + 1):
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
        print(f"[KVStorage::probe_max_prefix] {matched_pre_len=} for {len(key)=} with {min_length=} and {max_length=}")
        if matched_pre_len > min_length:
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
        matched_key: List[int],
        device: torch.device = torch.device("cuda"),
    ) -> torch.Tensor:
        # disk to cpu          
        kv_cpu_raw = self.db.get(self._make_key(matched_key))   
        # cpu reshape
        kv_np = np.frombuffer(kv_cpu_raw, dtype=np.float32).reshape(
            2, self.layer_num, len(matched_key), self.head_num, self.head_dim
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
        required_shape = (2, self.layer_num, kv_tensor.shape[2], self.head_num, self.head_dim)
        if kv_tensor.shape != required_shape:
            # print shapes
            print(f"[KVStorage::wait_for_kv] {required_shape=}")
            print(f"[KVStorage::wait_for_kv] {kv_tensor.shape=}")
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
        db_path="~/test_db",
    )
    
    key = [1, 2, 3, 4, 5]
    kv_tensor = torch.randn(2, layer_num, len(key), head_num, head_dim, dtype=torch.float16)
    
    kvs.put_prefix_kv(torch.tensor(key, dtype=torch.int32), kv_tensor)
    print(f"Stored kv_tensor with shape {kv_tensor.shape} for key {key}")
    matched_pre_len, kv_future = kvs.probe_max_prefix(
        torch.tensor(key, dtype=torch.int32), 
        min_length=0, 
        max_length=len(key)
    )
    fetched_kv_tensor = kvs.wait_for_kv(kv_future)
    
    assert matched_pre_len == len(key), "Matched prefix length does not match the original key length"
    assert torch.equal(fetched_kv_tensor.cpu(), kv_tensor.cpu()), "Fetched kv_tensor does not match the original kv_tensor"
    

    matched_pre_len, kv_future = kvs.probe_max_prefix(
        torch.tensor([1, 2, 3], dtype=torch.int32), 
        min_length=0, 
        max_length=3
    )
    fetched_kv_tensor = kvs.wait_for_kv(kv_future)
    assert matched_pre_len == 3, "Matched prefix length should be 3 for key [1, 2, 3]"
    assert torch.equal(fetched_kv_tensor.cpu(), kv_tensor.cpu()[:, :, :3, :, :]), "Fetched kv_tensor does not match the original kv_tensor for prefix [1, 2, 3]"