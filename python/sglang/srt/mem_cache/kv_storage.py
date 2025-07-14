import struct
import torch
import numpy as np
import rocksdb_binding as rocksdb
from typing import Tuple, Dict, Optional, List
import os
from concurrent.futures import ThreadPoolExecutor, Future, TimeoutError
from typing import Optional
import threading
from dataclasses import dataclass, field
import time
import queue

class KVStorage:
    _instance = None
    _lock = threading.Lock()

    @dataclass
    class Statistics:
        n_prefix_gets: int = 0
        t_prefix_get: float = 0.0 

        n_prefix_puts: int = 0
        t_prefix_put: float = 0.0   

        n_wait_for_kv: int = 0
        t_wait_for_kv: float = 0.0 

        n_executor_gets: int = 0
        t_executor_get: float = 0.0

        n_db_probes: int = 0
        t_db_probe: float = 0.0

        n_db_puts: int = 0
        t_db_put: float = 0.0

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
        executor_worker_num: int = 16,
        db_path: str = "db",
        compress: bool = True,
    ):
        print(f"[KVStorage::__init__] Initializing KVStorage with dtype={dtype}, kvtensor shape=({2}, {layer_num}, seq_len, {head_num}, {head_dim})")
        if getattr(self, '_initialized', False):
            return
        self._initialized = True

        self.dtype = dtype
        self.head_num = head_num
        self.head_dim = head_dim
        self.layer_num = layer_num
        self.db_path = os.path.expanduser(db_path)
        self.do_compress = compress

        self.db = rocksdb.RocksDB()
        print(f"Opening RocksDB at '{self.db_path}' with compression={self.do_compress}")
        open_status = self.db.open(self.db_path)
        assert open_status

        self.executor = ThreadPoolExecutor(max_workers=executor_worker_num)
        self.db_put_queue: queue.Queue[Tuple[List[int], torch.Tensor]] = queue.Queue(maxsize=20)
        threading.Thread(
            target=self._db_put_worker, daemon=True, name="DB Put Worker"
        ).start()
        self.statistics = self.Statistics()

    def statistics_str(self):
        return (
            f"[KVStorage] Statistics:\n"
            f"[Put] Prefix put count: {self.statistics.n_prefix_puts}, "
            f"Avg prefix put time: {self.statistics.t_prefix_put / max(1, self.statistics.n_prefix_puts):.6f} seconds\n"
            f"[Get] Prefix get count: {self.statistics.n_prefix_gets}, "
            f"Avg prefix get time: {self.statistics.t_prefix_get / max(1, self.statistics.n_prefix_gets):.6f} seconds\n"
            f"[Wait] Wait for KV count: {self.statistics.n_wait_for_kv}, "
            f"Avg wait time: {self.statistics.t_wait_for_kv / max(1, self.statistics.n_wait_for_kv):.6f} seconds\n"
            f"[Probe] DB probe count: {self.statistics.n_db_probes}, "
            f"Avg probe time: {self.statistics.t_db_probe / max(1, self.statistics.n_db_probes):.6f} seconds\n"
            f"[DB Put] Count: {self.statistics.n_db_puts}, "
            f"Avg time: {self.statistics.t_db_put / max(1, self.statistics.n_db_puts):.6f} seconds\n"
            f"[Executer Get] Count: {self.statistics.n_executor_gets}, "
            f"Avg time: {self.statistics.t_executor_get / max(1, self.statistics.n_executor_gets):.6f} seconds\n"
        )

    def _make_key(self, key: List[int]) -> bytes:
        assert isinstance(key, list), "Key must be a list of integers"
        assert isinstance(key[0], int), "List keys must contain integers"
        return np.array(key, dtype=np.int32).tobytes()

    def put_prefix_kv(
        self, 
        key: List[int],
        # shape: [2, layer_num, pre_len, head_num, head_dim]
        kv_tensor: torch.Tensor,
        block: bool = False,
    ):
        self.statistics.n_prefix_puts += 1
        start = time.perf_counter()
        required_shape = (2, self.layer_num, len(key), self.head_num, self.head_dim)
        assert kv_tensor.shape == required_shape, f"{kv_tensor.shape=} does not match {required_shape=}"
        if self.dtype not in [torch.float16, torch.float32, torch.float64]:
            kv_tensor = kv_tensor.to(torch.float32)
        if block:
            self._rocksdb_put(key, kv_tensor)
        else:
            try:
                self.db_put_queue.put((key, kv_tensor), block=False)
            except queue.Full:
                pass
        end = time.perf_counter()   
        self.statistics.t_prefix_put += (end - start)

    def _db_put_worker(self):
        while True:
            key, kv_tensor = self.db_put_queue.get()
            self.statistics.n_db_puts += 1
            start = time.perf_counter()

            key_bytes = self._make_key(key)
            exist_key_len = self._probe_max_prefix(
                key_bytes, min_length=0, max_length=len(key)
            )
            if exist_key_len == len(key):
                continue
            db_keys = [key_bytes[: (L + 1) * 4] for L in range(exist_key_len, len(key))]

            if self.do_compress:
                db_values = [
                    self.compress(kv_tensor[:, :, L, :, :])
                    for L in range(exist_key_len, len(key))
                ]
            else:
                db_values = [
                    kv_tensor[:, :, L, :, :].cpu().contiguous().numpy().data.tobytes()
                    for L in range(exist_key_len, len(key))
                ]

            self.db.batch_put(db_keys, db_values)
            end = time.perf_counter()
            self.statistics.t_db_put += end - start

    def compress(
        self,
        kv_tensor: torch.Tensor,  # shape: [2, layer_num, head_num, head_dim]
        num_bits: int = 4,
    ) -> bytes:
        data = kv_tensor.cpu().contiguous()

        group_dim = 1
        B: int = 2 ** num_bits - 1

        mn = torch.min(data, dim=group_dim, keepdim=True)[0]
        mx = torch.max(data, dim=group_dim, keepdim=True)[0]
        scale = B / (mx - mn + 1e-8)

        data = (data - mn) * scale
        data = data.clamp(0, B).round().to(torch.uint8)

        # Now pack data bits
        n_values_per_byte = 8 // num_bits
        flat = data.view(-1)

        # Pad to make multiple of n_values_per_byte
        remainder = flat.numel() % n_values_per_byte
        if remainder != 0:
            pad_size = n_values_per_byte - remainder
            flat = torch.cat([flat, torch.zeros(pad_size, dtype=flat.dtype)], dim=0)

        flat_np = flat.numpy()
        packed = bytearray()

        for i in range(0, len(flat_np), n_values_per_byte):
            byte = 0
            for j in range(n_values_per_byte):
                byte |= (flat_np[i + j] & B) << (j * num_bits)
            packed.append(byte)

        packed_data = bytes(packed)
        mn_bytes = mn.cpu().contiguous().numpy().tobytes()
        scale_bytes = scale.cpu().contiguous().numpy().tobytes()

        return packed_data + mn_bytes + scale_bytes

    def decompress(
        self,
        compressed: bytes,
        num_bits: int = 4,
        group_dim: int = 1,
    ) -> torch.Tensor:
        B = 2 ** num_bits - 1
        n_values_per_byte = 8 // num_bits
        
        # figure out sizes
        original_shape = (2, self.layer_num, self.head_num, self.head_dim)
        num_elements = np.prod(original_shape)
        num_packed_bytes = (num_elements + n_values_per_byte - 1) // n_values_per_byte

        # unpack data sections
        packed_data = compressed[:num_packed_bytes]
        rest = compressed[num_packed_bytes:]

        # compute sizes of mn, scale
        group_shape = list(original_shape)
        group_shape[group_dim] = 1
        num_groups = np.prod(group_shape)

        mn_bytes = rest[:self.dtype.itemsize * num_groups]
        scale_bytes = rest[self.dtype.itemsize * num_groups:]

        # decode mn and scale
        mn = torch.frombuffer(mn_bytes, dtype=self.dtype).reshape(group_shape)
        scale = torch.frombuffer(scale_bytes, dtype=self.dtype).reshape(group_shape)
        
        # unpack data values
        unpacked = np.zeros(num_elements, dtype=np.uint8)
        idx = 0
        for byte in packed_data:
            for j in range(n_values_per_byte):
                if idx >= num_elements:
                    break
                unpacked[idx] = (byte >> (j * num_bits)) & B
                idx += 1

        # reshape back
        data_q = torch.from_numpy(unpacked).view(original_shape).to(self.dtype)

        restored = data_q / scale + mn
        return restored


    def _probe_max_prefix(
        self,
        key: List[int] | bytes,
        min_length: int,
        max_length: int
    ) -> int:
        start = time.perf_counter()
        matched_pre_len = min_length
        if isinstance(key, list):
            key = self._make_key(key)
        # binary search for the longest prefix
        low, high = min_length, max_length
        while low < high:
            mid = (low + high + 1) // 2
            db_key = key[:mid * 4]
            exist = self.db.probe(db_key)
            self.statistics.n_db_probes += 1
            if exist:
                matched_pre_len = mid
                low = mid
            else:
                high = mid - 1
        end = time.perf_counter()
        self.statistics.t_db_probe += (end - start)
        return matched_pre_len

    def get_prefix_kv(
        self, 
        key: torch.Tensor, 
        min_length: int,
        max_length: int
    ) -> Tuple[int, Optional[Future]]:
        self.statistics.n_prefix_gets += 1
        start = time.perf_counter()
        matched_pre_len = self._probe_max_prefix(
            key,
            min_length=min_length,
            max_length=max_length
        )
        kv_future: Optional[Future] = None
        if matched_pre_len > min_length:
            matched_key = key[:matched_pre_len]
            # issue a worker thread to perform _rocksdb_get
            # the return value serves prefix [min_length, matched_pre_len]
            kv_future: Future = self.executor.submit(
                self._rocksdb_get,
                matched_key,
                min_length,
            )
        end = time.perf_counter()
        self.statistics.t_prefix_get += (end - start)
        return matched_pre_len, kv_future

    def _rocksdb_put(
        self,
        key: List[int],
        kv_tensor: torch.Tensor,
    ):
        kv_tensor = kv_tensor.cpu().numpy()
        exist_key_len = self._probe_max_prefix(
            key,
            min_length=0,
            max_length=len(key)
        ) 
        for L in range(exist_key_len, len(key)):
            prefix_ids = key[: L + 1]  # Prefix of length L
            prefix_tensor = kv_tensor[:, :, L, :, :]
            db_key = self._make_key(prefix_ids)
            value = prefix_tensor.cpu().numpy().tobytes()
            put_status = self.db.put(db_key, value)
            assert put_status

    def _rocksdb_get(
        self,
        matched_key: List[int],
        min_length: int,
    ) -> torch.Tensor:
        self.statistics.n_executor_gets += 1
        start = time.perf_counter()
        dtype = self.dtype if self.dtype in [torch.float16, torch.float32, torch.float64] else torch.float32
        matched_key_bytes = self._make_key(matched_key)
        db_keys = [matched_key_bytes[:(L + 1) * 4] for L in range(min_length, len(matched_key))]
        kv_cpu_raws = self.db.multiget(db_keys)
        kv_tensor = torch.stack(
            [
                self.decompress(kv_cpu_raws[db_key]) if self.do_compress else
                torch.frombuffer(
                    bytearray(kv_cpu_raws[db_key]),
                    dtype=dtype,
                    count=2 * self.layer_num * self.head_num * self.head_dim,
                ).reshape(2, self.layer_num, self.head_num, self.head_dim)
                for db_key in db_keys
            ],
            dim=2,
        ).to(self.dtype)
        end = time.perf_counter()
        self.statistics.t_executor_get += (end - start)
        return kv_tensor

    def wait_for_kv(
        self,
        kv_future: Future,
        timeout: Optional[float] = None,  # seconds; None = wait forever
    ) -> torch.Tensor:
        self.statistics.n_wait_for_kv += 1
        start = time.perf_counter()
        kv_tensor : torch.Tensor = kv_future.result(timeout=timeout)
        kv_tensor = kv_tensor.cuda()
        required_shape = (2, self.layer_num, kv_tensor.shape[2], self.head_num, self.head_dim)
        assert kv_tensor.shape == required_shape, f"{kv_tensor.shape=} does not match {required_shape=}"
        end = time.perf_counter()
        self.statistics.t_wait_for_kv += (end - start)
        return kv_tensor


if __name__ == "__main__":
    if os.path.exists(os.path.expanduser("~/test_db")):
        print("Deleting existing RocksDB at ~/test_db")
        os.system("rm -rf ~/test_db")
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
        compress=True
    )

    key = list(range(128))
    kv_tensor = torch.arange(
        2 * layer_num * len(key) * head_num * head_dim,
        dtype=torch.float16,
    ).reshape(2, layer_num, len(key), head_num, head_dim)
    kvs.put_prefix_kv(key, kv_tensor)
    time.sleep(2)
    print(f"Stored kv_tensor with shape {kv_tensor.shape} for {len(key)=}")
    matched_pre_len, kv_future = kvs.get_prefix_kv(
        key, 
        min_length=0, 
        max_length=len(key)
    )
    fetched_kv_tensor = kvs.wait_for_kv(kv_future)

    assert matched_pre_len == len(key), "Matched prefix length does not match the original key length"
    assert torch.allclose(fetched_kv_tensor.cpu(), kv_tensor.cpu(), rtol=1e-1), "Fetched kv_tensor does not match the original kv_tensor"

    matched_pre_len, kv_future = kvs.get_prefix_kv(
        key[:3], 
        min_length=0, 
        max_length=3
    )
    fetched_kv_tensor = kvs.wait_for_kv(kv_future)
    assert matched_pre_len == 3, "Matched prefix length should be 3 for key [1, 2, 3]"
    assert torch.allclose(fetched_kv_tensor.cpu(), kv_tensor.cpu()[:, :, :3, :, :], rtol=1e-1), "Fetched kv_tensor does not match the original kv_tensor for prefix [1, 2, 3]"

    print("=" * 40)
    print("KVStorage test passed successfully!")
    print(kvs.statistics_str())
