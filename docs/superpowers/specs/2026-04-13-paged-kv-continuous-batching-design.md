# Paged KV Physical Pool + Continuous Batching Design

## Goal

替换当前 per-sequence HF DynamicCache 方案，引入全局物理 KV 张量池 + 自定义 PagedLlamaAttention，最终实现 prefill 与 decode 序列在同一次 `model.forward()` 中处理（true continuous batching）。

## Architecture Overview

```
Phase 1（本 spec）：物理 KV 池 + PagedLlamaAttention，prefill / decode 仍分两次 forward
Phase 2（下一 spec）：varlen prefill + decode 合并为单次 forward（continuous batching）
```

**消除的开销**（vs 当前方案）：
- 每 decode step 重建 padded DynamicCache（left-pad + cat + split）
- per-sequence DynamicCache 对象碎片（B×32 层独立张量）
- HF `use_cache=True` 带来的 DynamicCache 分配

---

## File Structure

| 文件 | 动作 | 职责 |
|------|------|------|
| `nanopointllm/engine/kv_pool.py` | 新建 | 物理 KV 张量池 + slot 计算工具 |
| `nanopointllm/engine/forward_context.py` | 新建 | Thread-local ForwardContext（模式 + 路由信息）|
| `nanopointllm/llama/paged_attention.py` | 新建 | PagedLlamaAttention（store_kvcache + gather-sdpa）|
| `nanopointllm/llama/inject_paged.py` | 新建 | inject_paged_attention()：替换 32 层 LlamaAttention |
| `nanopointllm/engine/paged_model_runner.py` | 新建 | PagedModelRunner（替代 PointLLMModelRunner）|
| `nanopointllm/engine/llm_engine.py` | 修改 | 接入 PagedModelRunner + num_kv_blocks 参数 |
| `nanopointllm/engine/sequence.py` | 修改 | 移除 `past_key_values`，`kv_seq_len` 改用 block_table |
| `tests/test_store_kvcache.py` | 新建 | Triton kernel 正确性单测 |
| `tests/test_paged_attention.py` | 新建 | 单层 parity：PagedLlamaAttention vs HF LlamaAttention |
| `scripts/parity_paged_pointllm.py` | 新建 | 端到端 parity vs HF greedy（B=1/4/8/mixed）|
| `scripts/bench_paged_pointllm.py` | 新建 | decode bench 对比旧 engine |

---

## Component Designs

### 1. KVPool

全局物理 KV 张量，一次性分配，不随序列生命周期增删。

```python
# nanopointllm/engine/kv_pool.py

@dataclass
class KVPool:
    pool: torch.Tensor   # [2, num_layers, num_blocks, block_size, num_heads, head_dim]
    num_layers: int
    num_blocks: int
    block_size: int
    num_heads: int
    head_dim: int

    def k_cache(self, layer_idx: int) -> torch.Tensor:
        """[num_blocks, block_size, num_heads, head_dim]"""
        return self.pool[0, layer_idx]

    def v_cache(self, layer_idx: int) -> torch.Tensor:
        return self.pool[1, layer_idx]


def allocate_kv_pool(
    hf_model: nn.Module,
    num_blocks: int,
    block_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> KVPool:
    """
    从模型 config 读取 num_layers / num_heads / head_dim，
    分配 pool 张量并返回 KVPool。
    """
    cfg = hf_model.config
    num_layers = cfg.num_hidden_layers       # 32
    num_heads  = cfg.num_attention_heads     # 32
    head_dim   = cfg.hidden_size // num_heads  # 128
    pool = torch.zeros(
        2, num_layers, num_blocks, block_size, num_heads, head_dim,
        dtype=dtype, device=device,
    )
    return KVPool(pool, num_layers, num_blocks, block_size, num_heads, head_dim)
```

**Slot 计算规则：**
```
logical_pos = token index within sequence (0-based)
block_idx   = logical_pos // block_size
slot_offset = logical_pos  % block_size
physical_slot = block_table[block_idx] * block_size + slot_offset
```

---

### 2. ForwardContext

Thread-local 对象，在 `model.forward()` 前设置，在 `PagedLlamaAttention.forward()` 里读取。

```python
# nanopointllm/engine/forward_context.py

@dataclass
class ForwardContext:
    is_prefill: bool

    # 所有模式共用
    slot_mapping: torch.Tensor    # [total_new_tokens] int32，-1 = 跳过（padding）

    # prefill 模式：HF 的 attention_mask [B, max_len] 直接传入，用于 causal sdpa
    # decode 模式：
    block_tables: Optional[torch.Tensor]   # [B, max_num_blocks] int32
    context_lens: Optional[torch.Tensor]   # [B] int32


_ctx: threading.local = threading.local()

def set_forward_context(ctx: ForwardContext) -> None:
    _ctx.value = ctx

def get_forward_context() -> ForwardContext:
    return _ctx.value

def clear_forward_context() -> None:
    _ctx.value = None
```

---

### 3. Triton store_kvcache Kernel

将 K/V 写入物理池的对应 slot（参考 nano-vllm `layers/attention.py`，去掉 flash_attn 依赖）。

```python
# nanopointllm/llama/paged_attention.py（内含 kernel）

@triton.jit
def _store_kvcache_kernel(
    key_ptr, key_stride,        # [T, num_heads * head_dim]
    val_ptr, val_stride,
    k_cache_ptr, v_cache_ptr,   # [num_blocks, block_size, num_heads * head_dim]
    cache_stride,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot < 0:
        return                  # padding token，跳过
    offsets = tl.arange(0, D)
    k = tl.load(key_ptr + idx * key_stride + offsets)
    v = tl.load(val_ptr + idx * val_stride + offsets)
    tl.store(k_cache_ptr + slot * cache_stride + offsets, k)
    tl.store(v_cache_ptr + slot * cache_stride + offsets, v)


def store_kvcache(
    key: torch.Tensor,          # [T, H, D]  contiguous
    value: torch.Tensor,        # [T, H, D]
    k_cache: torch.Tensor,      # [num_blocks, block_size, H, D]
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor, # [T] int32
) -> None:
    T, H, D = key.shape
    HD = H * D
    key_flat   = key.reshape(T, HD)
    value_flat = value.reshape(T, HD)
    k_cache_flat = k_cache.view(-1, HD)
    v_cache_flat = v_cache.view(-1, HD)
    _store_kvcache_kernel[(T,)](
        key_flat, key_flat.stride(0),
        value_flat, value_flat.stride(0),
        k_cache_flat, v_cache_flat,
        k_cache_flat.stride(0),
        slot_mapping,
        D=HD,
    )
```

---

### 4. PagedLlamaAttention

替换 HF `LlamaAttention`，使用物理池读写 KV，两种模式：

- **prefill 模式**：store_kvcache 写池 + 用 HF 传入的 `attention_mask` 做 causal sdpa
- **decode 模式**：store_kvcache 写池 + gather block_tables + per-seq sdpa

```python
class PagedLlamaAttention(nn.Module):

    @classmethod
    def from_hf(
        cls,
        hf_attn,          # transformers LlamaAttention
        k_cache: torch.Tensor,   # [num_blocks, block_size, H, D] — pool slice for this layer
        v_cache: torch.Tensor,
    ) -> "PagedLlamaAttention":
        obj = cls.__new__(cls)
        nn.Module.__init__(obj)
        obj.config = hf_attn.config
        obj.layer_idx = hf_attn.layer_idx
        obj.head_dim = hf_attn.head_dim
        obj.num_heads = hf_attn.config.num_attention_heads
        obj.num_kv_heads = hf_attn.config.num_key_value_heads
        obj.num_kv_groups = hf_attn.num_key_value_groups
        obj.scaling = hf_attn.scaling
        obj.rotary_fn = hf_attn.rotary_fn       # apply_rotary_pos_emb function
        # Copy weight references (不复制权重，直接引用)
        obj.q_proj = hf_attn.q_proj
        obj.k_proj = hf_attn.k_proj
        obj.v_proj = hf_attn.v_proj
        obj.o_proj = hf_attn.o_proj
        # Physical KV cache slices
        obj.k_cache = k_cache
        obj.v_cache = v_cache
        return obj

    def forward(
        self,
        hidden_states: torch.Tensor,          # [B, seq_len, hidden]
        position_embeddings: tuple,           # (cos, sin) from LlamaRotaryEmbedding
        attention_mask: Optional[torch.Tensor],
        past_key_values=None,                 # ignored — physical pool handles KV
        **kwargs,
    ) -> tuple:
        ctx = get_forward_context()
        B, S, _ = hidden_states.shape
        H, D = self.num_heads, self.head_dim

        # QKV projection
        q = self.q_proj(hidden_states).reshape(B, S, H, D)
        k = self.k_proj(hidden_states).reshape(B, S, self.num_kv_heads, D)
        v = self.v_proj(hidden_states).reshape(B, S, self.num_kv_heads, D)

        # Apply RoPE (position_embeddings = (cos, sin) already computed by LlamaModel)
        cos, sin = position_embeddings
        q, k = self.rotary_fn(q, k, cos, sin, unsqueeze_dim=1)

        # Write new K/V to physical pool
        slot_map = ctx.slot_mapping  # [B*S] or [B] for decode
        store_kvcache(
            k.reshape(B * S, self.num_kv_heads, D).contiguous(),
            v.reshape(B * S, self.num_kv_heads, D).contiguous(),
            self.k_cache, self.v_cache, slot_map,
        )

        if ctx.is_prefill:
            # Dense causal attention with HF attention_mask (handles left-padding)
            # q: [B, H, S, D], k: [B, Hkv, S, D]
            q_t = q.transpose(1, 2)
            k_t = k.transpose(1, 2).repeat_interleave(self.num_kv_groups, dim=1)
            v_t = v.transpose(1, 2).repeat_interleave(self.num_kv_groups, dim=1)
            attn_out = F.scaled_dot_product_attention(
                q_t, k_t, v_t,
                attn_mask=attention_mask,
                is_causal=(attention_mask is None),
                scale=self.scaling,
            )  # [B, H, S, D]
            out = attn_out.transpose(1, 2).reshape(B, S, H * D)
        else:
            # Paged decode: gather K/V from block_tables, per-seq sdpa
            block_tables = ctx.block_tables   # [B, max_blocks]
            context_lens = ctx.context_lens   # [B]
            block_size = self.k_cache.shape[1]
            out_list = []
            for i in range(B):
                ctx_len = int(context_lens[i])
                num_blocks = (ctx_len + block_size - 1) // block_size
                bt = block_tables[i, :num_blocks]                   # [num_blocks]
                k_full = self.k_cache[bt].reshape(-1, self.num_kv_heads, D)[:ctx_len]  # [ctx_len, Hkv, D]
                v_full = self.v_cache[bt].reshape(-1, self.num_kv_heads, D)[:ctx_len]
                # q_i: [H, 1, D] → sdpa → [H, 1, D]
                qi = q[i].transpose(0, 1).unsqueeze(0)              # [1, H, 1, D]
                ki = k_full.unsqueeze(0).transpose(1, 2)            # [1, Hkv, ctx_len, D]
                vi = v_full.unsqueeze(0).transpose(1, 2)
                ki = ki.repeat_interleave(self.num_kv_groups, dim=1)
                vi = vi.repeat_interleave(self.num_kv_groups, dim=1)
                oi = F.scaled_dot_product_attention(qi, ki, vi, scale=self.scaling)
                out_list.append(oi.squeeze(0).transpose(0, 1))      # [1, H, D]
            out = torch.stack(out_list).reshape(B, 1, H * D)        # [B, 1, H*D]

        return self.o_proj(out), None   # (hidden_states, past_key_values=None)
```

---

### 5. inject_paged_attention()

```python
# nanopointllm/llama/inject_paged.py

def inject_paged_attention(hf_model: nn.Module, kv_pool: KVPool) -> None:
    """
    将 hf_model.model.layers[i].self_attn 全部替换为 PagedLlamaAttention。
    原地修改，不拷贝权重（引用共享）。
    """
    from transformers.models.llama.modeling_llama import LlamaAttention
    layers = hf_model.model.layers
    assert len(layers) == kv_pool.num_layers
    for layer_idx, layer in enumerate(layers):
        if isinstance(layer.self_attn, LlamaAttention):
            layer.self_attn = PagedLlamaAttention.from_hf(
                layer.self_attn,
                kv_pool.k_cache(layer_idx),
                kv_pool.v_cache(layer_idx),
            )
```

---

### 6. PagedModelRunner

```python
# nanopointllm/engine/paged_model_runner.py

class PagedModelRunner:

    def __init__(self, hf_model, kv_pool: KVPool, block_manager) -> None:
        self.hf_model = hf_model
        self.kv_pool = kv_pool
        self.block_manager = block_manager
        self.wrapper = PointLLMWrapper(hf_model)
        self.block_size = kv_pool.block_size

    # ── 点云编码（复用 PointLLMModelRunner 的逻辑）──────────────────────────
    def _encode_point_clouds_batch(self, seqs): ...  # 与当前实现相同

    # ── slot_mapping 计算 ────────────────────────────────────────────────
    def _compute_prefill_slot_mapping(
        self, seq: PointLLMSequence, max_len: int, pad_len: int,
    ) -> torch.Tensor:
        """
        返回 [max_len] int32 张量，pad 位置 = -1，真实 token 位置 = physical slot。
        """
        slots = torch.full((max_len,), -1, dtype=torch.int32)
        for pos in range(seq.num_tokens):
            blk = seq.block_table[pos // self.block_size]
            slots[pad_len + pos] = blk * self.block_size + (pos % self.block_size)
        return slots

    def _compute_decode_slot(self, seq: PointLLMSequence) -> int:
        """新 token 写入的物理 slot（decode 时 seq.num_tokens 已是新 token 后的长度）。"""
        pos = seq.num_tokens - 1
        blk = seq.block_table[pos // self.block_size]
        return blk * self.block_size + (pos % self.block_size)

    # ── prefill ──────────────────────────────────────────────────────────
    @torch.inference_mode()
    def run_prefill(self, seqs: list[PointLLMSequence]) -> list[int]:
        self._encode_point_clouds_batch(seqs)

        max_len = max(seq.num_tokens for seq in seqs)
        B = len(seqs)

        embeds_list, attn_masks, slot_maps = [], [], []
        for seq in seqs:
            L = seq.num_tokens
            pad_len = max_len - L
            if seq.inputs_embeds_cached is not None:
                emb = seq.inputs_embeds_cached
            else:
                device = next(self.wrapper.inner.parameters()).device
                ids = torch.tensor([seq.token_ids], dtype=torch.long, device=device)
                emb = self.wrapper.get_input_embeddings()(ids).squeeze(0)

            if pad_len > 0:
                emb = torch.cat([emb.new_zeros(pad_len, emb.shape[-1]), emb])
                mask = torch.cat([
                    torch.zeros(pad_len, dtype=torch.long, device=emb.device),
                    torch.ones(L, dtype=torch.long, device=emb.device),
                ])
            else:
                mask = torch.ones(L, dtype=torch.long, device=emb.device)

            embeds_list.append(emb)
            attn_masks.append(mask)
            slot_maps.append(
                self._compute_prefill_slot_mapping(seq, max_len, pad_len).to(emb.device)
            )

        inputs_embeds  = torch.stack(embeds_list)       # [B, max_len, H]
        attention_mask = torch.stack(attn_masks)         # [B, max_len]
        # slot_mapping: [B * max_len]
        slot_mapping = torch.stack(slot_maps).reshape(-1)

        set_forward_context(ForwardContext(
            is_prefill=True,
            slot_mapping=slot_mapping,
            block_tables=None,
            context_lens=None,
        ))
        try:
            dummy_ids = torch.zeros(B, max_len, dtype=torch.long, device=inputs_embeds.device)
            out = self.hf_model(
                input_ids=dummy_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                point_clouds=None,
                use_cache=False,
                return_dict=True,
            )
        finally:
            clear_forward_context()

        return out.logits[:, -1, :].float().argmax(dim=-1).tolist()

    # ── decode ───────────────────────────────────────────────────────────
    @torch.inference_mode()
    def run_decode(self, seqs: list[PointLLMSequence]) -> list[int]:
        B = len(seqs)
        device = next(self.hf_model.parameters()).device

        input_ids = torch.tensor([[seq.last_token] for seq in seqs],
                                 dtype=torch.long, device=device)
        positions  = torch.tensor([[seq.num_tokens - 1] for seq in seqs],
                                  dtype=torch.long, device=device)
        slot_mapping = torch.tensor(
            [self._compute_decode_slot(seq) for seq in seqs],
            dtype=torch.int32, device=device,
        )
        context_lens = torch.tensor(
            [seq.num_tokens for seq in seqs], dtype=torch.int32, device=device,
        )
        max_blocks = max(len(seq.block_table) for seq in seqs)
        block_tables = torch.tensor(
            [seq.block_table + [-1] * (max_blocks - len(seq.block_table)) for seq in seqs],
            dtype=torch.int32, device=device,
        )

        set_forward_context(ForwardContext(
            is_prefill=False,
            slot_mapping=slot_mapping,
            block_tables=block_tables,
            context_lens=context_lens,
        ))
        try:
            out = self.hf_model(
                input_ids=input_ids,
                position_ids=positions,
                attention_mask=None,
                use_cache=False,
                return_dict=True,
            )
        finally:
            clear_forward_context()

        logits = out.logits
        if logits.dim() == 3:
            logits = logits[:, -1, :]
        return logits.float().argmax(dim=-1).tolist()

    def run(self, seqs, is_prefill: bool) -> list[int]:
        if is_prefill:
            return self.run_prefill(seqs)
        return self.run_decode(seqs)
```

---

### 7. LLMEngine 修改

```python
# nanopointllm/engine/llm_engine.py（修改部分）

class PointLLMLLMEngine:
    def __init__(
        self,
        hf_model: nn.Module,
        eos_token_id: int,
        max_num_seqs: int = 8,
        max_num_batched_tokens: int = 4096,
        num_kv_blocks: int = 512,       # 新增：物理块数
        kvcache_block_size: int = 16,   # 默认 16 token/block
    ) -> None:
        device = next(hf_model.parameters()).device
        dtype  = next(hf_model.parameters()).dtype

        kv_pool = allocate_kv_pool(hf_model, num_kv_blocks, kvcache_block_size, device, dtype)
        block_manager = BlockManager(num_kv_blocks, kvcache_block_size)
        inject_paged_attention(hf_model, kv_pool)

        self.runner = PagedModelRunner(hf_model, kv_pool, block_manager)
        self.scheduler = Scheduler(
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
            eos_token_id=eos_token_id,
            block_manager=block_manager,
        )
```

---

### 8. Sequence 修改

移除 `past_key_values`，`kv_seq_len` 改为 `num_tokens`：

```python
# 删除
- self.past_key_values: Optional[Any] = None

# 修改
@property
def kv_seq_len(self) -> int:
    return self.num_tokens
```

---

## Data Flow

### Prefill

```
add_request(token_ids, point_clouds)
  → Scheduler.schedule() → prefill_seqs
  → BlockManager.allocate(seq)  [填充 seq.block_table]
  → PagedModelRunner.run_prefill(seqs)
      → encode_point_clouds_batch  → inputs_embeds_cached
      → compute slot_mapping (real tokens → physical slots, pads → -1)
      → set ForwardContext(is_prefill=True, slot_mapping)
      → hf_model.forward(inputs_embeds=..., use_cache=False)
          → LlamaModel → per-layer PagedLlamaAttention
              → QKV proj + RoPE
              → store_kvcache(K, V, k_cache, v_cache, slot_mapping)  [Triton]
              → causal sdpa(Q, K, V, attention_mask)
          → lm_head → logits
      → clear ForwardContext
  → greedy sample → first generated token
  → Scheduler.postprocess
```

### Decode

```
Scheduler.schedule() → decode_seqs
  → BlockManager.may_append(seq)  [必要时分配新 block]
  → PagedModelRunner.run_decode(seqs)
      → input_ids = [[last_token] for each seq]
      → compute slot_mapping [B] = physical slot for new token
      → build block_tables [B, max_blocks], context_lens [B]
      → set ForwardContext(is_prefill=False, slot_mapping, block_tables, context_lens)
      → hf_model.forward(input_ids=..., position_ids=..., use_cache=False)
          → per-layer PagedLlamaAttention
              → QKV proj + RoPE
              → store_kvcache(new K, new V, slot)  [Triton]
              → for each seq: gather k_cache[block_table], sdpa(q, k_full, v_full)
          → lm_head → logits [B, 1, V]
      → greedy sample
  → Scheduler.postprocess
```

---

## Testing Strategy

### 单元测试（不需要真实权重）

**`tests/test_store_kvcache.py`**
- 构造 key/value [T, H, D]，slot_mapping [T]，k_cache/v_cache [num_blocks, block_size, H, D]
- 调用 `store_kvcache`
- 断言 k_cache[slot//block_size, slot%block_size] == key[t]
- 覆盖：slot=-1（padding）跳过，slot=0 边界，连续 slot，非连续 slot

**`tests/test_paged_attention.py`**
- 单层 PagedLlamaAttention vs HF LlamaAttention parity
- prefill 模式：[B=2, seq_len=8, H=4096]，构造 causal mask
- decode 模式：[B=2, 1 new token]，固定 block_table，验证 gather 结果

### 集成测试（需要真实权重 + GPU）

**`scripts/parity_paged_pointllm.py`**
- 与 `parity_engine_pointllm.py` 同结构，但使用 `PointLLMLLMEngine(num_kv_blocks=512)`
- 4 cases：B=1（1 pc）/ B=4（4 pc）/ B=8（8 pc）/ mixed（2 pc + 2 text-only）
- 与 `hf_greedy_generate` 对比，全部 PASS 方视为验收通过

**`scripts/bench_paged_pointllm.py`**
- 对比 paged engine vs 旧 padded-KV engine 的 decode latency + tokens/sec
- Batch sizes: 1, 2, 4, 8

---

## Phase 2 Preview：Varlen Continuous Batching

阶段二在本设计基础上做以下扩展：

1. **ForwardContext 增加 varlen 字段**：`cu_seqlens_q`, `cu_seqlens_k`, `max_seqlen_q`, `max_seqlen_k`
2. **PagedLlamaAttention 增加 varlen prefill 路径**：将 B 个 prefill 序列的 token 拼成 `[total_tokens, H, D]`，用 varlen causal attention（naive 实现 or flash_attn_varlen 若可用）
3. **PagedModelRunner 合并 forward**：单次 `model.forward()` 同时处理 prefill tokens + decode tokens（分别 cat inputs，分别提取 logits）
4. **Scheduler 取消 is_prefill 二态**：允许同一 step 里有 prefill 和 decode 序列

阶段二独立 spec，依赖阶段一验收。

---

## Out of Scope（Phase 1）

- flash_attn_with_kvcache（环境未安装）
- CUDA graph capture
- Tensor parallelism
- Prefix cache 跨请求复用（BlockManager 元数据已有，物理 KV 复用待 Phase 2）
- 动态扩容 KV pool（`num_kv_blocks` 在引擎创建时固定）
