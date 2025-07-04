import struct
import torch
import numpy as np
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

        self.db = SimpleFileDB(db_path=db_path)
        
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
            self.db.put(key, value)

    def probe_max_prefix(
        self,
        key: torch.Tensor, 
        prefix_len_rt: int,
        min_length: int,
        max_length: int
    ) -> Tuple[Optional[List[int]], int, Optional[Future]]:
        matched_pre_len = min_length
        print("[KVStorage::probe_max_prefix] key is ", key)
        for pre_len in range(min_length + 1, max_length + 1):
            position = self.db.probe(self._make_key(key[:pre_len]))
            if position is None:
                break
            else:
                matched_pre_len = pre_len
        
        if matched_pre_len > min_length:
            # issue a worker thread to perform _rocksdb_get
            kv_future: Future = _executor.submit(
                self._db_io,
                position,
                prefix_len_rt,
                matched_pre_len,
                torch.device("cuda")
            )
                
            return matched_pre_len, kv_future
        else:
            return matched_pre_len, None

    def _db_io(
        self,
        position,
        prefix_len_rt: int,
        prefix_len_kvs: int,
        device: torch.device = torch.device("cuda"),
    ) -> torch.Tensor:
        # disk to cpu
        result = self.db.get(position=position)                
        kv_cpu_raw = result.data
        # cpu reshape
        kv_np = np.frombuffer(kv_cpu_raw[prefix_len_rt, :], dtype=np.float32).reshape(
            2, self.layer_num, prefix_len_kvs - prefix_len_rt, self.head_num, self.head_dim
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

class SimpleFileDB:
    """
    A simple file-based key-value store with three basic operations:
    put, get, and probe. It operates directly on byte strings for keys and values.

    - `put`: Appends a new (key, value) pair to the end of the file.
    - `get`: Retrieves a (key, value) pair from a specific line number (position).
    - `probe`: Searches for a key and returns its line number (position) if found.

    WARNING: This is a simplistic implementation for debugging and is not
             performant or robust for production use.
    """
    def __init__(self, db_path: str = "~/simple_file_db.bin"):
        self.db_path = os.path.expanduser(db_path)
        self.file_lock = threading.Lock()

        if not os.path.exists(self.db_path):
            with open(self.db_path, 'wb') as f:
                pass
        
        print(f"SimpleFileDB (Binary) initialized. Using file at: '{self.db_path}'")

    def put(self, key: bytes, value: bytes) -> None:
        """
        Appends a new key-value record to the binary file.

        Args:
            key (bytes): The key.
            value (bytes): The value.
        """
        key_len = len(key)
        value_len = len(value)
        
        # 'Q' represents unsigned long long (8 bytes)
        packed_key_len = struct.pack('Q', key_len)
        packed_value_len = struct.pack('Q', value_len)

        with self.file_lock:
            # 使用 'ab' (append binary) 模式
            with open(self.db_path, 'ab') as f:
                f.write(packed_key_len)
                f.write(key)
                f.write(packed_value_len)
                f.write(value)

    def get(self, position: int) -> Optional[Tuple[bytes, bytes]]:
        """
        Retrieves the key-value pair starting at a specific file offset (position).

        Args:
            position (int): The file offset to start reading from.

        Returns:
            A tuple of (key, value) in bytes, or None if reading fails.
        """
        if position < 0:
            return None

        with self.file_lock:
            with open(self.db_path, 'rb') as f:
                f.seek(position)
                
                # read the length of key (8 bytes)
                packed_key_len = f.read(8)
                if len(packed_key_len) < 8: return None
                key_len = struct.unpack('Q', packed_key_len)[0]
                
                # read key
                key = f.read(key_len)
                if len(key) < key_len: return None

                # read the length of value (8 bytes)
                packed_value_len = f.read(8)
                if len(packed_value_len) < 8: return None
                value_len = struct.unpack('Q', packed_value_len)[0]
                
                # read value
                value = f.read(value_len)
                if len(value) < value_len: return None

                return key, value

    def probe(self, target_key: bytes) -> Optional[int]:
        """
        Searches for a target key in the file and returns its starting file offset (position).
        This implementation iterates through the file record by record.

        Args:
            target_key (bytes): The key to search for.

        Returns:
            The file offset (position) if the key is found, otherwise None.
        """
        with self.file_lock:
            with open(self.db_path, 'rb') as f:
                current_pos = 0
                while True:
                    record_start_pos = f.tell()
                    packed_key_len = f.read(8)
                    if not packed_key_len:
                        break  # file end
                    key_len = struct.unpack('Q', packed_key_len)[0]
                    key = f.read(key_len)

                    packed_value_len = f.read(8)
                    value_len = struct.unpack('Q', packed_value_len)[0]
                    
                    if key == target_key:
                        return record_start_pos

                    # 跳过 value 部分，移动到下一条记录
                    f.seek(value_len, 1) # 1 表示从当前位置向后移动
        return None
    