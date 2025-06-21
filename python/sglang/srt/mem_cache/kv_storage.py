import struct
import torch
import numpy as np
import rocksdb_binding as rocksdb
from typing import Tuple
import os


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
        self, dtype: torch.dtype, head_num: int, head_dim: int, layer_num: int
    ):
        self.dtype = dtype
        self.head_num = head_num
        self.head_dim = head_dim
        self.layer_num = layer_num

        self.db = rocksdb.RocksDB()
        if not self.db.open(os.path.expanduser("~/LSMCache_db")):
            raise RuntimeError("Failed to open RocksDB for KV storage.")

    def _make_key(self, input_ids: torch.Tensor, layer_id: int) -> bytes:
        """Prefix bytes + layer_id (little-endian 32-bit) → RocksDB key"""
        return input_ids.cpu().numpy().tobytes() + struct.pack("<i", layer_id)

    def put_prefix(
        self, input_ids: torch.Tensor, kv_tensor: torch.Tensor, layer_id: int
    ):
        """Store KV tensor for one layer & prefix into RocksDB"""
        key = self._make_key(input_ids, layer_id)
        value = kv_tensor.to(torch.float32).cpu().numpy().tobytes()
        self.db.put(key, value)

    def get_prefix(
        self, input_ids: torch.Tensor, device: torch.device = torch.device("cuda")
    ) -> Tuple[torch.Tensor, int, torch.Tensor]:

        max_len = input_ids.shape[0]
        prefix_len = 0
        matched_prefix = None

        # ── find longest prefix via layer‑0 ─────────────────────────────
        for L in range(max_len, 0, -1):
            cand = input_ids[:L]
            val = self.db.get(self._make_key(cand, layer_id=0))
            if not val == "":
                prefix_len = L
                matched_prefix = cand
                break

        if matched_prefix is None:
            return None, 0, None

        kv_buffer = torch.empty(
            (self.layer_num, prefix_len, self.head_num, self.head_dim),
            dtype=self.dtype,
            device=device,
        )

        for layer_id in range(self.layer_num):
            raw = self.db.get(self._make_key(matched_prefix, layer_id))
            if raw == "":
                raise KeyError(
                    f"Missing KV for prefix (len={prefix_len}) at layer {layer_id}"
                )
            kv_np = np.frombuffer(raw, dtype=np.float32).reshape(
                prefix_len, self.head_num, self.head_dim
            )
            kv_buffer[layer_id] = torch.as_tensor(
                kv_np, dtype=self.dtype, device=device
            )

        return matched_prefix, prefix_len, kv_buffer

    def get_kv_cache(self, layer_id: int, kv_loc: torch.Tensor) -> torch.Tensor:
        return kv_loc[layer_id]
