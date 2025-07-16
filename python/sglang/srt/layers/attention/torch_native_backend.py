from __future__ import annotations

from typing import TYPE_CHECKING, Optional, List

import torch
from torch.nn.functional import scaled_dot_product_attention

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner


class TorchNativeAttnBackend(AttentionBackend):
    def __init__(self, model_runner: ModelRunner):
        super().__init__()

        self.forward_metadata = None
        self.device = model_runner.device
        
        self.enable_kvstore = model_runner.enable_kvstore
        self.kvstore = model_runner.kvstore
        
        self.layer_num = model_runner.num_effective_layers

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """Init the metadata for a forward pass."""
        pass

    def _run_sdpa_forward_extend(
        self,
        query: torch.Tensor,
        output: torch.Tensor,
        k_cache: torch.Tensor,              # forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
        v_cache: torch.Tensor,              # forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id)
        req_to_token: torch.Tensor,         # forward_batch.req_to_token_pool.req_to_token
        req_pool_indices: torch.Tensor,     # forward_batch.req_pool_indices
        seq_lens: torch.Tensor,             # forward_batch.seq_lens
        extend_seq_lens: torch.Tensor,      # forward_batch.extend_seq_lens
        extend_prefix_lens: Optional[torch.Tensor] = None,   # forward_batch.extend_prefix_lens / schedule_batch.prefix_lens
        prefix_lens_kvs: Optional[List[int]] = None,
        scaling=None,                       # layer.scaling
        enable_gqa=False,                   # use_gqa
        causal=False,                       # not layer.is_cross_attention or layer.attn_type == AttentionType.ENCODER_ONLY
    ):
        """Run the extend forward by using torch native sdpa op.

        Args:
            query: [num_tokens, num_heads, head_size]
            output: [num_tokens, num_heads, head_size]
            k_cache: [max_total_num_tokens, num_heads, head_size]
            v_cache: [max_total_num_tokens, num_heads, head_size]
            req_to_token: [max_num_reqs, max_context_len]
            req_pool_indices: [num_seqs]
            seq_lens: [num_seqs]
            extend_prefix_lens: [num_seqs]
            extend_seq_lens: [num_seqs]
            prefix_lens_kvs: [num_seqs]
            scaling: float or None
            enable_gqa: bool
            causal: bool

        Returns:
            output: [num_tokens, num_heads, head_size]
        """
        if self.kvstore is None:
            assert seq_lens.shape[0] == extend_prefix_lens.shape[0]
        else:
            assert seq_lens.shape[0] == len(prefix_lens_kvs)
        assert seq_lens.shape[0] == extend_seq_lens.shape[0]

        # [num_tokens, num_heads, head_size] -> [num_heads, num_tokens, head_size]
        query = query.movedim(0, query.dim() - 2)

        start_q, start_kv = 0, 0
        for seq_idx in range(seq_lens.shape[0]):
            # TODO: this loop process a sequence per iter, this is inefficient.
            # Need optimize the performance later.

            extend_seq_len_q = extend_seq_lens[seq_idx]
            if self.kvstore is None:
                prefill_seq_len_q = extend_prefix_lens[seq_idx]
            else:
                prefill_seq_len_q = prefix_lens_kvs[seq_idx]

            seq_len_kv = seq_lens[seq_idx]
            end_q = start_q + extend_seq_len_q
            end_kv = start_kv + seq_len_kv
            
            # print(f"seq_idx={seq_idx}, seq_len_kv={seq_len_kv}, extend_seq_len_q={extend_seq_len_q}, start_q={start_q}, end_q={end_q}")

            per_req_query = query[:, start_q:end_q, :]
            per_req_query_redudant = torch.empty(
                (per_req_query.shape[0], seq_len_kv, per_req_query.shape[2]),
                dtype=per_req_query.dtype,
                device=per_req_query.device,
            )

            # print(f"shape of per_req_query_redudant[:, prefill_seq_len_q:, :]: {per_req_query_redudant[:, prefill_seq_len_q:, :].shape}")
            # print(f"shape of per_req_query: {per_req_query.shape}")

            per_req_query_redudant[:, prefill_seq_len_q:, :] = per_req_query

            # get key and value from cache. per_req_tokens contains the kv cache
            # index for each token in the sequence.
            req_pool_idx = req_pool_indices[seq_idx]
            per_req_tokens = req_to_token[req_pool_idx, :seq_len_kv]
            
            per_req_key = k_cache[per_req_tokens].movedim(0, query.dim() - 2)
            per_req_value = v_cache[per_req_tokens].movedim(0, query.dim() - 2)

            per_req_out_redudant = (
                scaled_dot_product_attention(
                    per_req_query_redudant.unsqueeze(0),
                    per_req_key.unsqueeze(0),
                    per_req_value.unsqueeze(0),
                    enable_gqa=enable_gqa,
                    scale=scaling,
                    is_causal=causal,
                )
                .squeeze(0)
                .movedim(query.dim() - 2, 0)
            )
            output[start_q:end_q, :, :] = per_req_out_redudant[prefill_seq_len_q:, :, :]
            start_q, start_kv = end_q, end_kv
        return output

    def _run_sdpa_forward_decode(
        self,
        query: torch.Tensor,
        output: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        req_to_token: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        scaling=None,
        enable_gqa=False,
        causal=False,
    ):
        """Run the decode forward by using torch native sdpa op.

        Args:
            query: [num_tokens, num_heads, head_size]
            output: [num_tokens, num_heads, head_size]
            k_cache: [max_total_num_tokens, num_heads, head_size]
            v_cache: [max_total_num_tokens, num_heads, head_size]
            req_to_token: [max_num_reqs, max_context_len]
            req_pool_indices: [num_seqs]
            seq_lens: [num_seqs]
            scaling: float or None
            enable_gqa: bool
            causal: bool

        Returns:
            output: [num_tokens, num_heads, head_size]
        """

        # [num_tokens, num_heads, head_size] -> [num_heads, num_tokens, head_size]
        query = query.movedim(0, query.dim() - 2)

        start_q, start_kv = 0, 0
        for seq_idx in range(seq_lens.shape[0]):
            # TODO: this loop process a sequence per iter, this is inefficient.
            # Need optimize the performance later.

            seq_len_q = 1
            seq_len_kv = seq_lens[seq_idx]
            end_q = start_q + seq_len_q
            end_kv = start_kv + seq_len_kv

            per_req_query = query[:, start_q:end_q, :]

            # get key and value from cache. per_req_tokens contains the kv cache
            # index for each token in the sequence.
            req_pool_idx = req_pool_indices[seq_idx]
            per_req_tokens = req_to_token[req_pool_idx, :seq_len_kv]
            per_req_key = k_cache[per_req_tokens].movedim(0, query.dim() - 2)
            per_req_value = v_cache[per_req_tokens].movedim(0, query.dim() - 2)

            per_req_out = (
                scaled_dot_product_attention(
                    per_req_query.unsqueeze(0),
                    per_req_key.unsqueeze(0),
                    per_req_value.unsqueeze(0),
                    enable_gqa=enable_gqa,
                    scale=scaling,
                    is_causal=causal,
                )
                .squeeze(0)
                .movedim(query.dim() - 2, 0)
            )
            output[start_q:end_q, :, :] = per_req_out
            start_q, start_kv = end_q, end_kv

        return output

    def forward_extend(
        self,
        q,
        k,
        v,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
    ):
        if layer.qk_head_dim != layer.v_head_dim:
            o = q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))
        else:
            o = torch.empty_like(q)
        
        if save_kv_cache:
            if self.enable_kvstore:
                loc_idx = 0
                for i, kv_future in enumerate(forward_batch.kv_futures):
                    if kv_future is not None:
                        assert forward_batch.prefix_lens_extra[i] > 0, \
                            f"kv_future is not None but {forward_batch.prefix_lens_extra[i]=}"
                        fetched_kv = self.kvstore.wait_for_kv(kv_future)  # [2, layer_num, prefix_len, head_num, head_dim]

                        k_fetched = fetched_kv[0, layer.layer_id]  # [prefix_len, head_num, head_dim]
                        v_fetched = fetched_kv[1, layer.layer_id]
                        assert k_fetched.shape == v_fetched.shape, \
                            f"fetched kv shape mismatch: {k_fetched.shape}, {v_fetched.shape}"
                        
                        # TODO: Optimize ineffictive IO bandwidth
                        k_fetched_effictive = k_fetched[-forward_batch.prefix_lens_extra[i]:]
                        v_fetched_effictive = v_fetched[-forward_batch.prefix_lens_extra[i]:]
                            
                        forward_batch.token_to_kv_pool.set_kv_buffer(
                            layer,
                            forward_batch.out_cache_loc_for_kvstore[loc_idx: loc_idx + forward_batch.prefix_lens_extra[i]],
                            k_fetched_effictive,
                            v_fetched_effictive,
                        )
                        loc_idx += forward_batch.prefix_lens_extra[i]
                    else:
                        assert forward_batch.prefix_lens_extra[i] == 0, \
                            f"kv_future is None but prefix_lens_extra[{i}] = {forward_batch.prefix_lens_extra[i]}"
                forward_batch.token_to_kv_pool.set_kv_buffer(
                    layer, forward_batch.out_cache_loc_for_recompute, k, v
                )
            # if self.enable_kvstore and sum(forward_batch.prefix_lens_extra) > 0:
            #     forward_batch.token_to_kv_pool.set_kv_buffer(
            #         layer, forward_batch.out_cache_loc[:-sum(forward_batch.prefix_lens_extra)], k, v
            #     )
            else: 
                forward_batch.token_to_kv_pool.set_kv_buffer(
                    layer, forward_batch.out_cache_loc, k, v
                )
        
        use_gqa = layer.tp_q_head_num != layer.tp_k_head_num

        q_ = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
        o_ = o.view(-1, layer.tp_q_head_num, layer.v_head_dim)

        causal = True
        if layer.is_cross_attention or layer.attn_type == AttentionType.ENCODER_ONLY:
            causal = False

        if self.kvstore is None:
            self._run_sdpa_forward_extend(
                query=q_,
                output=o_,
                k_cache=forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id),
                v_cache=forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id),
                req_to_token=forward_batch.req_to_token_pool.req_to_token,
                req_pool_indices=forward_batch.req_pool_indices,
                seq_lens=forward_batch.seq_lens,
                extend_prefix_lens=forward_batch.extend_prefix_lens,
                extend_seq_lens=forward_batch.extend_seq_lens,
                scaling=layer.scaling,
                enable_gqa=use_gqa,
                causal=causal,
            )
        else:
            self._run_sdpa_forward_extend(
                query=q_,
                output=o_,
                k_cache=forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id),
                v_cache=forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id),
                req_to_token=forward_batch.req_to_token_pool.req_to_token,
                req_pool_indices=forward_batch.req_pool_indices,
                seq_lens=forward_batch.seq_lens,
                prefix_lens_kvs=forward_batch.prefix_lens_kvs,
                extend_seq_lens=forward_batch.extend_seq_lens,
                scaling=layer.scaling,
                enable_gqa=use_gqa,
                causal=causal,
            )
        
        return o

    def forward_decode(
        self,
        q,
        k,
        v,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
    ):
        # During torch.compile, there is a bug in rotary_emb that causes the
        # output value to have a 3D tensor shape. This reshapes the output correctly.
        q = q.reshape(-1, layer.tp_q_head_num * layer.qk_head_dim)

        if layer.qk_head_dim != layer.v_head_dim:
            o = q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))
        else:
            o = torch.empty_like(q)

        if save_kv_cache:
            forward_batch.token_to_kv_pool.set_kv_buffer(
                layer, forward_batch.out_cache_loc, k, v
            )

        use_gqa = layer.tp_q_head_num != layer.tp_k_head_num

        q_ = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
        o_ = o.view(-1, layer.tp_q_head_num, layer.v_head_dim)

        self._run_sdpa_forward_decode(
            query=q_,
            output=o_,
            k_cache=forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id),
            v_cache=forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id),
            req_to_token=forward_batch.req_to_token_pool.req_to_token,
            req_pool_indices=forward_batch.req_pool_indices,
            seq_lens=forward_batch.seq_lens,
            scaling=layer.scaling,
            enable_gqa=use_gqa,
            causal=False,
        )

        return o

    def support_triton(self):
        return False
