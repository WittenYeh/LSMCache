from sglang.srt.mem_cache.kv_storage import KVStorage
import torch

head_num = 2
head_dim = 4
layer_num = 8

if __name__ == "__main__":
    kvs = KVStorage(
        dtype=torch.float16,
        head_num=head_num,
        head_dim=head_dim,
        layer_num=layer_num,
        db_path="~/LSMCache/KVS",
    )
    
    prefix_1 = torch.tensor([1, 2, 3, 4, 5], dtype=torch.int32)
    prefix_2 = torch.tensor([1, 2, 3], dtype=torch.int32)
    
    # put prefix 1
    original_kv_tensors = []
    for layer_id in range(layer_num):
        kv_tensor = torch.randn(
            (prefix_1.shape[0], head_num, head_dim), dtype=torch.float16
        )
        original_kv_tensors.append(kv_tensor)
        kvs.put_prefix(prefix_1, kv_tensor, layer_id)
        
    # retrive prefix 1
    matched_prefix, prefix_len, kv_buffer = kvs.get_prefix(prefix_1)
    retrived_kv_tensors = [kvs.get_kv_cache(layer_id, kv_buffer).cpu() for layer_id in range(layer_num)]
    
    # print results
    print(f"Matched Prefix: {matched_prefix}")
    print(f"Prefix Length: {prefix_len}")
    assert torch.equal(matched_prefix, prefix_1), "Matched prefix does not match original prefix"
    assert prefix_len == prefix_1.shape[0], "Prefix length does not match original prefix length"
    assert all(torch.equal(original_kv_tensors[layer_id], retrived_kv_tensors[layer_id]) for layer_id in range(layer_num)), "Retrieved KV tensors do not match original KV tensors"
    
    # lookfor prefix 2
    matched_prefix, prefix_len, kv_buffer = kvs.get_prefix(prefix_2)
    
    # print results
    print(f"Matched Prefix: {matched_prefix}")
    print(f"Prefix Length: {prefix_len}")
    assert prefix_len == 3, "Prefix length does not match expected length"