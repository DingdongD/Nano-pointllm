# Paged KV Phase 2: Varlen Continuous Batching Design

## Goal

Replace the current batch-mode scheduler (all seqs prefill together, then all decode together) with true continuous batching: new requests enter the running set as soon as KV pool space is available, and every `step()` issues exactly **one** `model.forward()` over a flat `[1, total_tokens, H]` tensor covering both new prefill tokens and decode tokens from existing sequences.

## Architecture

### Data flow per `step()`

```
scheduler.schedule()
  → (prefill_seqs=[A,B], decode_seqs=[C,D,E])

runner.run_mixed(prefill_seqs, decode_seqs)
  1. Point-cloud encoding for A, B
  2. Build inputs_embeds [1, total_tokens, H]
       [A.new_tokens | B.new_tokens | C.last | D.last | E.last]
  3. Build ForwardContext
       slot_mapping   [total_tokens]          (-1 for cached-prefix positions)
       block_tables   [5, max_blocks]
       context_lens   [5]                     full KV length per seq
       seq_lens       [lenA_new, lenB_new, 1, 1, 1]
       position_ids   [1, total_tokens]
  4. model.forward(inputs_embeds=[1, total_tokens, H])   ← single forward
       LlamaModel: rotary → 32 × (LayerNorm + PagedLlamaAttention + FFN)
       PagedLlamaAttention: store ALL K/V → per-seq SDPA from pool
  5. Extract logits at last-token position of each seq → token_ids [5]

scheduler.postprocess(prefill_seqs + decode_seqs, token_ids)
```

### Files changed (5 total)

| File | Change |
|------|--------|
| `nanopointllm/engine/forward_context.py` | Remove `is_prefill`, add `seq_lens: list[int]` |
| `nanopointllm/engine/scheduler.py` | `schedule()` → `(prefill_seqs, decode_seqs)` |
| `nanopointllm/engine/llm_engine.py` | `step()` calls `run_mixed` |
| `nanopointllm/engine/paged_model_runner.py` | Add `run_mixed`, replace `run` |
| `nanopointllm/llama/paged_attention.py` | `forward()` → varlen per-seq loop |

---

## Module Specifications

### 1. ForwardContext (`engine/forward_context.py`)

```python
@dataclass
class ForwardContext:
    slot_mapping:  torch.Tensor   # [total_tokens] int32; -1 = skip (cached prefix)
    block_tables:  torch.Tensor   # [total_seqs, max_blocks] int32
    context_lens:  torch.Tensor   # [total_seqs] int32 — full KV length per seq
    position_ids:  torch.Tensor   # [1, total_tokens] int64
    seq_lens:      list[int]      # new query tokens per seq (prompt_new or 1)
```

`is_prefill` is removed. `PagedLlamaAttention` infers prefill vs decode from `seq_lens[i] > 1`.

The LlamaModel forward-pre patch (`_install_position_ids_patch`) continues to inject `position_ids` from context into `LlamaModel.forward()` kwargs — no changes needed there.

---

### 2. Scheduler (`engine/scheduler.py`)

```python
def schedule(self) -> tuple[list[PointLLMSequence], list[PointLLMSequence]]:
    """
    Returns (prefill_seqs, decode_seqs).

    Prefill seqs: admitted from waiting this step.
      - block_manager.can_allocate(seq) must pass
      - token budget: sum(seq.num_tokens - seq.num_cached_tokens) <= max_num_batched_tokens
      - count: total running seqs <= max_num_seqs
    Decode seqs: all currently running seqs not being prefilled this step.
    Both lists may be empty simultaneously only when engine is finished.
    """
    bm = self.block_manager
    prefill_seqs: list[PointLLMSequence] = []
    token_budget = 0

    while self.waiting:
        seq = self.waiting[0]
        if len(self.running) + len(prefill_seqs) >= self.max_num_seqs:
            break
        if bm is not None and not bm.can_allocate(seq):
            break
        new_tokens = seq.num_tokens - seq.num_cached_tokens
        if token_budget + new_tokens > self.max_num_batched_tokens:
            break
        token_budget += new_tokens
        self.waiting.popleft()
        if bm is not None:
            bm.allocate(seq)
        seq.status = SequenceStatus.RUNNING
        self.running.append(seq)
        prefill_seqs.append(seq)

    prefill_set = set(id(s) for s in prefill_seqs)
    decode_seqs = [s for s in self.running if id(s) not in prefill_set]

    if bm is not None:
        for seq in decode_seqs:
            if bm.can_append(seq):
                bm.may_append(seq)

    return prefill_seqs, decode_seqs
```

`postprocess` is unchanged: receives `prefill_seqs + decode_seqs` and corresponding `token_ids`.

`is_finished()` is unchanged: `not self.waiting and not self.running`.

---

### 3. PagedLlamaAttention varlen forward (`llama/paged_attention.py`)

```python
def forward(self, hidden_states, position_embeddings, attention_mask=None, **kwargs):
    ctx = get_forward_context()
    _, S, _ = hidden_states.shape   # shape is [1, total_tokens, hidden]
    H, Hkv, D = self.num_heads, self.num_kv_heads, self.head_dim

    # QKV projections — single pass over total_tokens
    q = self.q_proj(hidden_states).view(1, S, H,   D).transpose(1, 2)  # [1,H,S,D]
    k = self.k_proj(hidden_states).view(1, S, Hkv, D).transpose(1, 2)
    v = self.v_proj(hidden_states).view(1, S, Hkv, D).transpose(1, 2)

    cos, sin = position_embeddings
    q, k = self.rotary_fn(q, k, cos, sin)

    # Write all new K/V to physical pool (slot=-1 positions are skipped)
    k_store = k.transpose(1, 2).reshape(S, Hkv, D).contiguous()
    v_store = v.transpose(1, 2).reshape(S, Hkv, D).contiguous()
    store_kvcache(k_store, v_store, self.k_cache, self.v_cache, ctx.slot_mapping)

    # Per-sequence SDPA reading full K/V history from pool
    block_size = self.k_cache.shape[1]
    seq_outs, q_offset = [], 0
    for i, q_len in enumerate(ctx.seq_lens):
        ctx_len  = int(ctx.context_lens[i])
        num_blks = (ctx_len + block_size - 1) // block_size
        bt       = ctx.block_tables[i, :num_blks]                         # valid blocks only
        k_full   = self.k_cache[bt].reshape(-1, Hkv, D)[:ctx_len]        # [ctx_len, Hkv, D]
        v_full   = self.v_cache[bt].reshape(-1, Hkv, D)[:ctx_len]

        qi = q[0, :, q_offset:q_offset+q_len, :].unsqueeze(0)            # [1, H, q_len, D]
        ki = k_full.permute(1,0,2).unsqueeze(0).repeat_interleave(self.num_kv_groups, 1)
        vi = v_full.permute(1,0,2).unsqueeze(0).repeat_interleave(self.num_kv_groups, 1)

        oi = F.scaled_dot_product_attention(
            qi, ki, vi,
            is_causal=(q_len > 1),  # True for prefill, False for decode
            scale=self.scaling,
        )  # [1, H, q_len, D]
        seq_outs.append(oi.squeeze(0).permute(1, 0, 2).reshape(q_len, H * D))
        q_offset += q_len

    out = torch.cat(seq_outs).unsqueeze(0)   # [1, total_tokens, H*D]
    return self.o_proj(out), None
```

Key invariants:
- `slot_mapping` has `-1` for cached-prefix positions; `store_kvcache` skips them
- `context_lens[i]` covers full KV including prefix cache; pool gather reads complete history
- `is_causal=(q_len > 1)` is correct: a single decode token never needs a causal mask

---

### 4. PagedModelRunner.run_mixed (`engine/paged_model_runner.py`)

```python
@torch.inference_mode()
def run_mixed(
    self,
    prefill_seqs: list[PointLLMSequence],
    decode_seqs: list[PointLLMSequence],
) -> list[int]:
    """
    Single model.forward() over prefill new-tokens + decode last-tokens.
    Returns one next token_id per sequence (prefill_seqs first, then decode_seqs).
    """
    all_seqs = prefill_seqs + decode_seqs
    if not all_seqs:
        return []

    self._encode_point_clouds_batch(prefill_seqs)
    device = next(self.hf_model.parameters()).device

    emb_parts, pos_parts, slot_parts = [], [], []
    seq_lens_list, context_lens_list, block_tables_list = [], [], []

    for seq in prefill_seqs:
        new_start = seq.num_cached_tokens
        new_len   = seq.num_tokens - new_start
        seq_lens_list.append(new_len)
        context_lens_list.append(seq.num_tokens)
        block_tables_list.append(seq.block_table)

        # Embedding: only new tokens (prefix already in pool)
        if seq.inputs_embeds_cached is not None:
            emb = seq.inputs_embeds_cached[new_start:]
        else:
            ids = torch.tensor(seq.token_ids[new_start:], dtype=torch.long, device=device)
            emb = self.wrapper.get_input_embeddings()(ids.unsqueeze(0)).squeeze(0)
        emb_parts.append(emb)

        # Positions: [new_start .. num_tokens-1]
        pos_parts.append(torch.arange(new_start, seq.num_tokens,
                                      dtype=torch.long, device=device))

        # Slots: cached positions → -1, new positions → physical slot
        slots = []
        for abs_pos in range(new_start, seq.num_tokens):
            blk = abs_pos // self.block_size
            off = abs_pos % self.block_size
            slots.append(seq.block_table[blk] * self.block_size + off)
        slot_parts.append(torch.tensor(slots, dtype=torch.int32, device=device))

    for seq in decode_seqs:
        seq_lens_list.append(1)
        context_lens_list.append(seq.num_tokens)
        block_tables_list.append(seq.block_table)

        ids = torch.tensor([seq.last_token], dtype=torch.long, device=device)
        emb_parts.append(self.wrapper.get_input_embeddings()(ids.unsqueeze(0)).squeeze(0))

        pos = seq.num_tokens - 1
        slot = seq.block_table[pos // self.block_size] * self.block_size + pos % self.block_size
        pos_parts.append(torch.tensor([pos],  dtype=torch.long,  device=device))
        slot_parts.append(torch.tensor([slot], dtype=torch.int32, device=device))

    total_tokens  = sum(seq_lens_list)
    inputs_embeds = torch.cat(emb_parts).unsqueeze(0)            # [1, T, H]
    slot_mapping  = torch.cat(slot_parts)                        # [T]
    position_ids  = torch.cat(pos_parts).unsqueeze(0)            # [1, T]
    max_blocks    = max(len(bt) for bt in block_tables_list)
    block_tables  = torch.tensor(
        [bt + [-1] * (max_blocks - len(bt)) for bt in block_tables_list],
        dtype=torch.int32, device=device,
    )
    context_lens  = torch.tensor(context_lens_list, dtype=torch.int32, device=device)

    dummy_ids = torch.zeros(1, total_tokens, dtype=torch.long, device=device)
    set_forward_context(ForwardContext(
        slot_mapping=slot_mapping,
        block_tables=block_tables,
        context_lens=context_lens,
        position_ids=position_ids,
        seq_lens=seq_lens_list,
    ))
    try:
        out = self.hf_model(
            input_ids=dummy_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=None,
            point_clouds=None,
            use_cache=False,
            return_dict=True,
        )
    finally:
        clear_forward_context()

    logits = out.logits  # [1, total_tokens, vocab]
    last_positions = [sum(seq_lens_list[:i+1]) - 1 for i in range(len(all_seqs))]
    idx = torch.tensor(last_positions, dtype=torch.long, device=device)
    return logits[0, idx, :].float().argmax(dim=-1).tolist()
```

---

### 5. LLMEngine.step() (`engine/llm_engine.py`)

```python
def step(self) -> None:
    prefill_seqs, decode_seqs = self.scheduler.schedule()
    if not prefill_seqs and not decode_seqs:
        return
    token_ids = self.runner.run_mixed(prefill_seqs, decode_seqs)
    self.scheduler.postprocess(prefill_seqs + decode_seqs, token_ids)
```

`generate()`, `add_request()`, `is_finished()` are unchanged.

---

## Scheduling Policy

**Memory pressure (KV pool full):** Block-and-wait. If `can_allocate(seq)` returns False for the first waiting sequence, no new sequences are admitted this step. Running decode sequences continue unaffected.

**Token budget:** `max_num_batched_tokens` limits the total number of **new** prefill tokens admitted per step (not counting prefix-cached tokens). Decode tokens (1 per seq) are not counted against this budget.

**Ordering:** Prefill sequences always appear before decode sequences in `all_seqs`. Within each group, FIFO order is preserved.

---

## Testing Strategy

### New unit tests (`tests/test_varlen_paged.py`)

**`test_varlen_attention_single_prefill`**
Single prefill seq, verify output matches old `run_prefill` path token-for-token.

**`test_varlen_attention_single_decode`**
Single decode seq (seq with 1 new token), verify output matches old `run_decode` path.

**`test_varlen_attention_mixed`**
1 prefill seq + 1 decode seq in one step. Verify each seq's predicted token matches running them separately.

**`test_continuous_batching_scheduler`**
Submit 4 requests, run 2 decode steps, then submit 4 more. Verify new requests start processing in the next step after admission, not after the original 4 finish.

### Updated integration test (`scripts/parity_paged_pointllm.py`)

Add case `continuous_8` (B=4 running, B=4 arrive after 2 steps) and verify all 8 sequences produce the same output as running them in two independent batches.

### Updated benchmark (`scripts/bench_paged_pointllm.py`)

Add `--arrival_interval` flag. When set, submits `batch_size` requests, waits `arrival_interval` decode steps, submits another `batch_size`. Reports:
- `time_to_first_token_ms` per request
- Total throughput (tokens/sec)

Compare continuous vs batch mode.

---

## Out of Scope

- Flash attention varlen (`flash_attn_varlen_func` — not installed)
- CUDA graph capture
- Preemption / swap-to-CPU
- Tensor parallelism
- Batched padded decode attention (orthogonal optimization, can be added later without design change)
