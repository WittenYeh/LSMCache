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
        n_prefix_puts: int = 0

        n_db_gets: int = 0
        n_db_puts: int = 0

        t_get: float = 0.0 
        t_put: float = 0.0 

        n_wait_for_kv: int = 0
        t_wait_for_kv: float = 0.0 

        n_executer_gets: int = 0
        t_executer_get: float = 0.0

        n_db_probes: int = 0
        t_db_probe: float = 0.0

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
        self.put_queue_gpu: queue.Queue[Tuple[List[int], torch.Tensor]] = queue.Queue(maxsize=100)
        self.put_queue_cpu: queue.Queue[Tuple[bytes, torch.Tensor]] = queue.Queue()
        threading.Thread(
            target=self._gpu_to_cpu_put_worker,
            daemon=True,
            name="GPU to CPU Put Worker"
        ).start()
        threading.Thread(
            target=self._cpu_to_db_put_worker,
            daemon=True,
            name="CPU to DB Put Worker"
        ).start()

        self.statistics = self.Statistics()

    def statistics_str(self):
        return (
            f"[KVStorage] Statistics:\n"
            f"[Put] Prefix put count: {self.statistics.n_prefix_puts}, "
            f"DB puts count: {self.statistics.n_db_puts}, "
            f"Avg prefix put time: {self.statistics.t_put / max(1, self.statistics.n_prefix_puts):.6f} seconds\n"
            f"[Get] Prefix get count: {self.statistics.n_prefix_gets}, "
            f"DB gets count: {self.statistics.n_db_gets}, "
            f"Avg prefix get time: {self.statistics.t_get / max(1, self.statistics.n_prefix_gets):.6f} seconds\n"
            f"[Wait] Wait for KV count: {self.statistics.n_wait_for_kv}, "
            f"Avg wait time: {self.statistics.t_wait_for_kv / max(1, self.statistics.n_wait_for_kv):.6f} seconds\n"
            f"[Probe] DB probe count: {self.statistics.n_db_probes}, "
            f"Avg probe time: {self.statistics.t_db_probe / max(1, self.statistics.n_db_probes):.6f} seconds\n"
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
                self.put_queue_gpu.put((key, kv_tensor), block=False)
            except queue.Full:
                pass
        end = time.perf_counter()   
        self.statistics.t_put += (end - start)

    def _gpu_to_cpu_put_worker(self):
        while True:
            key, kv_tensor = self.put_queue_gpu.get()
            kv_tensor = kv_tensor.cpu()
            key_bytes = self._make_key(key)
            self.put_queue_cpu.put((key_bytes, kv_tensor))

    def _cpu_to_db_put_worker(self):
        while True:
            key_bytes, kv_tensor = self.put_queue_cpu.get()
            exist_key_len = self._probe_max_prefix(
                key_bytes,
                min_length=0,
                max_length=len(key_bytes) // 4,  # Each int32 is 4 bytes
            )
            for L in range(exist_key_len, len(key_bytes) // 4):
                db_key = key_bytes[: (L + 1) * 4]
                if self.do_compress:
                    value = self.compress(kv_tensor[:, :, L, :, :])
                else:
                    value = kv_tensor[:, :, L, :, :].numpy().tobytes()
                put_status = self.db.put(db_key, value)
                assert put_status, f"Failed to put {len(db_key)=} into RocksDB"
                self.statistics.n_db_puts += 1

    def compress(
        self,
        tensor: torch.Tensor,
        num_bits: int = 8,
    ):
        assert num_bits <= 8
        group_dim = 1
        new_shape = torch.Size([2, 1, self.layer_num, self.head_num, self.head_dim])
        data = tensor.view(new_shape)

        B : int = 2 ** num_bits - 1
        mn = torch.min(data, dim=group_dim + 1, keepdim=True)[0]
        mx = torch.max(data, dim=group_dim + 1, keepdim=True)[0]

        scale = B / (mx - mn)
        data = data - mn
        data.mul_(scale)

        data = data.clamp_(0, B).round_().to(torch.uint8)
        # pack data, mn, scale
        data = data.numpy().tobytes()
        mn = mn.numpy().tobytes()
        scale = scale.numpy().tobytes()
        return data + mn + scale

    def decompress(self, data_bytes: bytes):
        data_shape = torch.Size([2, 1, self.layer_num, self.head_num, self.head_dim])
        mn_shape = torch.Size([2, 1, 1, self.head_num, self.head_dim])
        scale_shape = torch.Size([2, 1, 1, self.head_num, self.head_dim])

        data_numel = data_shape.numel()
        mn_numel = mn_shape.numel()
        scale_numel = scale_shape.numel()

        data = torch.frombuffer(bytearray(data_bytes), dtype=torch.uint8, count=data_numel, offset=0)
        mn = torch.frombuffer(bytearray(data_bytes), dtype=self.dtype, count=mn_numel, offset=data_numel)
        scale = torch.frombuffer(bytearray(data_bytes), dtype=self.dtype, count=scale_numel, offset=data_numel + mn_numel * self.dtype.itemsize)

        data = data.reshape(data_shape).to(self.dtype)
        mn = mn.reshape(mn_shape)
        scale = scale.reshape(scale_shape)

        # Dequantize
        scale_inv = 1.0 / scale
        reconstructed = data * scale_inv + mn

        return reconstructed.squeeze(1)

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
        for pre_len in range(max_length, min_length, -1):
            db_key = key[:pre_len * 4]  # Each int32 is 4 bytes
            exist = self.db.probe(db_key)
            self.statistics.n_db_probes += 1
            if exist:
                matched_pre_len = pre_len
                break
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

        if matched_pre_len > min_length:
            matched_key = key[:matched_pre_len]

            self.statistics.n_executer_gets += 1
            # issue a worker thread to perform _rocksdb_get
            # the return value serves prefix [min_length, matched_pre_len]
            kv_future: Future = self.executor.submit(
                self._rocksdb_get,
                matched_key,
                min_length,
                torch.device("cuda")
            )
            end = time.perf_counter()
            self.statistics.t_get += (end - start)
            return matched_pre_len, kv_future
        else:
            end = time.perf_counter()
            self.statistics.t_get += (end - start)
            return matched_pre_len, None

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
            value = prefix_tensor.tobytes()
            put_status = self.db.put(db_key, value)
            self.statistics.n_db_puts += 1
            assert put_status

    def _rocksdb_get(
        self,
        matched_key: List[int],
        min_length: int,
        device: torch.device = torch.device("cuda"),
    ) -> torch.Tensor:
        self.statistics.n_db_gets += 1
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
        )

        return kv_tensor.to(self.dtype).to(device)

    def wait_for_kv(
        self,
        kv_future: Future,
        timeout: Optional[float] = None,  # seconds; None = wait forever
    ) -> torch.Tensor:
        self.statistics.n_wait_for_kv += 1
        start = time.perf_counter()
        kv_tensor : torch.Tensor = kv_future.result(timeout=timeout)
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
    )

    db_value = torch.arange(
        2 * layer_num * head_num * head_dim,
        dtype=torch.float16,
    ).reshape(2, layer_num, head_num, head_dim)
    compressed_value = kvs.compress(db_value, num_bits=8)
    decompressed_value = kvs.decompress(compressed_value)
    assert torch.allclose(db_value, decompressed_value, rtol=1e-1), "Decompressed value does not match original value"

    kvs.do_compress = False
    key = list(range(256))
    kv_tensor = torch.arange(
        2 * layer_num * len(key) * head_num * head_dim,
        dtype=torch.float16,
    ).reshape(2, layer_num, len(key), head_num, head_dim)
    kvs.put_prefix_kv(key, kv_tensor, block=True)
    print(f"Stored kv_tensor with shape {kv_tensor.shape} for {len(key)=}")
    matched_pre_len, kv_future = kvs.get_prefix_kv(
        key, 
        min_length=0, 
        max_length=len(key)
    )
    fetched_kv_tensor = kvs.wait_for_kv(kv_future)

    assert matched_pre_len == len(key), "Matched prefix length does not match the original key length"
    assert torch.equal(fetched_kv_tensor.cpu(), kv_tensor.cpu()), "Fetched kv_tensor does not match the original kv_tensor"

    matched_pre_len, kv_future = kvs.get_prefix_kv(
        key[:3], 
        min_length=0, 
        max_length=3
    )
    fetched_kv_tensor = kvs.wait_for_kv(kv_future)
    assert matched_pre_len == 3, "Matched prefix length should be 3 for key [1, 2, 3]"
    assert torch.equal(fetched_kv_tensor.cpu(), kv_tensor.cpu()[:, :, :3, :, :]), "Fetched kv_tensor does not match the original kv_tensor for prefix [1, 2, 3]"

    print("=" * 40)
    print("KVStorage test passed successfully!")
    print(kvs.statistics_str())
