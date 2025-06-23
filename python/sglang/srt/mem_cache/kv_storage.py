import struct
import torch
import numpy as np
import rocksdb_binding as rocksdb
from typing import Tuple
import os
from concurrent.futures import ThreadPoolExecutor, Future, TimeoutError
from typing import Optional

_executor = ThreadPoolExecutor(max_workers=4)


class KVStorage:
    """RocksDB-backed KV cache

    Key format  =   <prefix bytes>  ||  <layer_id:int32>
    * prefix = raw `input_ids.tobytes()` (variable length)
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
        db_path: str = "~/LSMCache_db",
    ):
        self.dtype = dtype
        self.head_num = head_num
        self.head_dim = head_dim
        self.layer_num = layer_num

        self.db = rocksdb.RocksDB()
        if not self.db.open(os.path.expanduser(db_path)):
            raise RuntimeError("Failed to open RocksDB for KV storage.")

    def _make_key(self, input_ids: torch.Tensor, layer_id: int) -> bytes:
        """Prefix bytes + layer_id (little-endian 32-bit) → RocksDB key"""
        return input_ids.cpu().numpy().tobytes() + struct.pack("<i", layer_id)

    def put_prefix(
        self, input_ids: torch.Tensor, kv_tensor: torch.Tensor, layer_id: int
    ):
        if kv_tensor.shape[0] != input_ids.shape[0]:
            raise ValueError("kv_tensor first dim must equal len(input_ids)")

        total_tokens = input_ids.shape[0]
        for L in range(1, total_tokens + 1):
            prefix_ids = input_ids[:L]  # Prefix of length L
            prefix_kv = kv_tensor[:L]  # Corresponding KV
            key = self._make_key(prefix_ids, layer_id)
            value = prefix_kv.to(torch.float32).cpu().numpy().tobytes()
            self.db.put(key, value)

    def _fetch_and_stage(
        self, keys, layer_num, prefix_len, head_num, head_dim, dtype, device
    ):
        """Runs inside a worker thread."""
        # 1) RocksDB multiget
        values = self.db.multiget(keys)

        # 2) pinned CPU buffer
        cpu_buf = torch.empty(
            (layer_num, prefix_len, head_num, head_dim),
            dtype=torch.float32,
            pin_memory=True,
        )

        # 3) disk to CPU 
        for layer_id, key in enumerate(keys):
            raw = values.get(key)
            if raw is None:
                raise KeyError(f"Missing KV for layer {layer_id}")
            kv_np = np.frombuffer(raw, dtype=np.float32).reshape(
                prefix_len, head_num, head_dim
            )
            np.copyto(cpu_buf[layer_id].numpy(), kv_np)

        # 4) cpu to GPU
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            gpu_buf = cpu_buf.to(device=device, dtype=dtype, non_blocking=True)

        # ensure the copy finishes before Future returns the tensor
        stream.synchronize()
        return gpu_buf

    def get_prefix(
        self,
        input_ids: torch.Tensor,
        device: torch.device = torch.device("cuda"),
    ) -> Tuple[Optional[torch.Tensor], int, Optional[Future]]:
        max_len = input_ids.shape[0]
        matched_prefix, prefix_len = None, 0

        # find longest prefix by probing layer-0
        for L in range(max_len, 0, -1):
            cand = input_ids[:L]
            if self.db.get(self._make_key(cand, 0)) is not None:
                matched_prefix, prefix_len = cand, L
                break

        if matched_prefix is None:
            return None, 0, None  # nothing cached

        # build keys for all layers
        keys = [self._make_key(matched_prefix, lid) for lid in range(self.layer_num)]

        # submit non-blocking task
        kv_future: Future = _executor.submit(
            self._fetch_and_stage,
            keys,
            self.layer_num,
            prefix_len,
            self.head_num,
            self.head_dim,
            self.dtype,
            device,
        )

        # immediate return – Future will carry the GPU buffer later
        return matched_prefix, prefix_len, kv_future

    def get_kv_cache(self, layer_id: int, kv_loc: Future) -> Optional[torch.Tensor]:
        if not kv_loc.done():
            return None
        gpu_buf = kv_loc.result()
        return gpu_buf[layer_id]

    def get_kv_cache_blocking(
        self,
        layer_id: int,
        kv_loc: Future,
        timeout: Optional[float] = None,  # seconds; None = wait forever
    ) -> torch.Tensor:
        gpu_buf = kv_loc.result(timeout=timeout)
        return gpu_buf[layer_id]
