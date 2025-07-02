import struct
import torch
import numpy as np
import pyrocksdb
from typing import Tuple, Dict, Optional, List
import os
from concurrent.futures import ThreadPoolExecutor, Future, TimeoutError
from typing import Optional
import threading

_executor = ThreadPoolExecutor(max_workers=4)


class KVStorage:
    """RocksDB-backed KV cache

    Key format  =   <prefix bytes>  ||  <layer_id:int32>
    * prefix = raw `key.tobytes()` (variable length)
    * layer  = little-endian int32 appended at the end

    `get_prefix()` scans for the longest matching prefix *only in layer-0*.
    Once found, it loads corresponding KV tensors from *all layers*
    into a single contiguous GPU buffer (returned as `kv_loc`).
    Later, `get_kv_cache()` slices this buffer using the `layer_id`.
    """

    def __init__(
        self,
        dtype: torch.dtype,
        head_num: int,
        head_dim: int,
        layer_num: int,
        executor_worker_num: int = 4,
        db_path: str = "~/LSMCache_db",
    ):
        self.dtype = dtype
        self.head_num = head_num
        self.head_dim = head_dim
        self.layer_num = layer_num
        self.db_path = db_path

        self.db = pyrocksdb.DB()
        self.open_opts = pyrocksdb.Options(create_if_missing=True)
        self.write_opts = pyrocksdb.WriteOptions()
        self.read_opts = pyrocksdb.ReadOptions()
        
        open_status = self.db.open(self.open_opts, self.db_path)
        assert(open_status.ok())
        
        self.executor = ThreadPoolExecutor(max_workers=executor_worker_num)
        
    def __del__(self):
        if hasattr(self, 'db'):
            del self.db
            self.db = None
            print(f"RocksDB instance at '{self.db_path}' closed.")
        
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)

    def _make_key(self, key: torch.Tensor) -> bytes:
        """Prefix bytes → RocksDB key"""
        return key.cpu().numpy().tobytes()

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
            key = self._make_key(prefix_ids)
            value = prefix_tensor.to(torch.float32).cpu().numpy().tobytes()
            put_status = self.db.put(key, value)
            assert put_status.ok()

    def probe_max_prefix(
        self, 
        key: torch.Tensor, 
        min_length: int,
        max_length: int
    ) -> Tuple[Optional[List[int]], int, Optional[Future]]:
        matched_pre_len = min_length
        for pre_len in range(min_length + 1, max_length + 1):
            result = self.db.get(self.read_opts, self._make_key(key[:, pre_len]))
            if result is None:
                break
            else:
                matched_pre_len = pre_len
        
        if matched_pre_len != 0:
            matched_key = key[:, matched_pre_len]
            
            # issue a worker thread to perform _rocksdb_get
            kv_future: Future = _executor.submit(
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
        result = self.db.get(self.read_opts, self._make_key())                
        kv_cpu_raw = result.data
        # cpu reshape
        kv_np = np.frombuffer(kv_cpu_raw, dtype=np.float32).reshape(
            2, self.layer_num, matched_key.shape[0], self.head_num, self.head_dim
        )
        kv_tensor = torch.from_numpy(kv_np)
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
        