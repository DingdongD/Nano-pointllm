#!/usr/bin/env python3
"""
PointLLM prefill pipeline timing breakdown:
  FPS | KNN | Local Encoder | reduce_dim+pos_embed | PointBERT Transformer | norm+pool
  | point_proj | prepare_inputs_embeds | LLM forward
"""
from __future__ import annotations
import sys, os, time
import torch

_POINTLLM = os.environ.get("POINTLLM_ROOT", "/home/PointLLM")
if _POINTLLM not in sys.path:
    sys.path.insert(0, _POINTLLM)

from transformers import AutoTokenizer
from pointllm.model import PointLLMLlamaForCausalLM
from pointllm.conversation import conv_templates
from pointllm.model.pointbert import misc as pb_misc
from pointllm.model.pointbert.dvae import knn_point

from nanopointllm.parity.model_path import validate_pretrained_local_or_hub
from nanopointllm.parity.attn_stability import set_eager_attention_if_supported
from nanopointllm.engine.hf_hybrid import hf_prefill

MODEL_PATH = "/mnt/llm_data/pointllm_ckpt/PointLLM_7B_v1.2"
DEVICE = torch.device("cuda:1")
DTYPE = torch.bfloat16
WARMUP = 3
RUNS = 5
B_LIST = [1, 4, 8]


# ── cuda timer helper ────────────────────────────────────────────────────────

def cuda_ms(fn):
    """Run fn() WARMUP+RUNS times, return mean wall time in ms (last RUNS)."""
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(RUNS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    return sum(times) / len(times)


# ── load model ───────────────────────────────────────────────────────────────

def load_model():
    path = validate_pretrained_local_or_hub(MODEL_PATH)
    tok = AutoTokenizer.from_pretrained(path, use_fast=False)
    model = PointLLMLlamaForCausalLM.from_pretrained(
        path, torch_dtype=DTYPE, low_cpu_mem_usage=True, use_cache=True,
    ).to(DEVICE)
    model.initialize_tokenizer_point_backbone_config_wo_embedding(tok)
    model.eval()
    set_eager_attention_if_supported(model)
    return model, tok


# ── prompt helpers ───────────────────────────────────────────────────────────

def build_input_ids(tok, model):
    conv = conv_templates["vicuna_v1_1"].copy()
    cfg = model.get_model().point_backbone_config
    pt_len = cfg["point_token_len"]
    patch = cfg["default_point_patch_token"]
    if cfg["mm_use_point_start_end"]:
        s, e = cfg["default_point_start_token"], cfg["default_point_end_token"]
        qs = s + patch * pt_len + e + "\nWhat is this?"
    else:
        qs = patch * pt_len + "\nWhat is this?"
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    enc = tok([conv.get_prompt()], return_tensors="pt")
    return enc["input_ids"].to(DEVICE), enc["attention_mask"].to(DEVICE)


# ── breakdown ────────────────────────────────────────────────────────────────

@torch.inference_mode()
def breakdown_single(model, input_ids, attention_mask):
    """Per-sequence timing breakdown for B=1."""
    inner = model.get_model()
    backbone = inner.point_backbone
    proj = inner.point_proj
    gd = backbone.group_divider

    pc_full = torch.randn(1, 8192, 6, device=DEVICE, dtype=DTYPE)
    xyz = pc_full[:, :, :3]   # [1, 8192, 3] — only xyz used by FPS/KNN
    pc_rgb = pc_full           # full 6-dim input

    print("\n=== Single-sequence (B=1) breakdown ===")

    # 1. FPS only
    def _fps():
        pb_misc.fps(xyz, gd.num_group)

    fps_ms = cuda_ms(_fps)
    centers = pb_misc.fps(xyz, gd.num_group)
    print(f"  FPS  ({gd.num_group} groups from {8192} pts, Python loop):  {fps_ms:7.2f} ms")

    # 2. KNN only (needs center)
    def _knn():
        knn_point(gd.group_size, xyz, centers)

    knn_ms = cuda_ms(_knn)
    print(f"  KNN  ({gd.group_size} neighbors per group):                 {knn_ms:7.2f} ms")

    # 3. group_divider full (FPS + KNN + index gather)
    def _group():
        gd(pc_rgb)

    group_ms = cuda_ms(_group)
    print(f"  Group (FPS+KNN+gather total):                     {group_ms:7.2f} ms")

    neighborhood, center_out = gd(pc_rgb)

    # 4. Local encoder (Encoder MLP on each patch)
    enc_module = backbone.encoder

    def _encoder():
        enc_module(neighborhood)

    enc_ms = cuda_ms(_encoder)
    group_tokens = enc_module(neighborhood)
    print(f"  Local Encoder (patch-level MLP):                  {enc_ms:7.2f} ms")

    # 5. reduce_dim + pos_embed (linear projections)
    def _reduce_pos():
        t = backbone.reduce_dim(group_tokens)
        backbone.pos_embed(center_out)
        return t

    reduce_ms = cuda_ms(_reduce_pos)
    reduced = backbone.reduce_dim(group_tokens)
    pos = backbone.pos_embed(center_out)
    cls_tok = backbone.cls_token.expand(1, -1, -1)
    cls_pos = backbone.cls_pos.expand(1, -1, -1)
    x = torch.cat([cls_tok, reduced], dim=1)
    pos_full = torch.cat([cls_pos, pos], dim=1)
    print(f"  reduce_dim + pos_embed:                           {reduce_ms:7.2f} ms")

    # 6. PointBERT Transformer blocks (depth=2)
    def _transformer():
        backbone.blocks(x, pos_full)

    trans_ms = cuda_ms(_transformer)
    x_out = backbone.blocks(x, pos_full)
    print(f"  PointBERT Transformer ({backbone.depth} blocks):              {trans_ms:7.2f} ms")

    # 7. norm + max pool
    def _norm_pool():
        xn = backbone.norm(x_out)
        torch.cat([xn[:, 0], xn[:, 1:].max(1)[0]], dim=-1).unsqueeze(1)

    pool_ms = cuda_ms(_norm_pool)
    xn = backbone.norm(x_out)
    feat = torch.cat([xn[:, 0], xn[:, 1:].max(1)[0]], dim=-1).unsqueeze(1)
    print(f"  Norm + max pool:                                  {pool_ms:7.2f} ms")

    # 8. Full backbone (group_divider → blocks → norm+pool)
    def _backbone():
        backbone(pc_rgb)

    bb_ms = cuda_ms(_backbone)
    raw = backbone(pc_rgb)
    print(f"  ─── Backbone total:                               {bb_ms:7.2f} ms")

    # 9. point_proj (linear: backbone_dim → LLM hidden_size)
    def _proj():
        proj(raw[0])

    proj_ms = cuda_ms(_proj)
    feat_proj = proj(raw[0])
    print(f"  point_proj (linear → LLM hidden):                 {proj_ms:7.2f} ms")

    # 10. prepare_inputs_embeds (text embed + fusion)
    cfg = inner.point_backbone_config
    use_start_end = cfg.get("mm_use_point_start_end", False)
    patch_id = cfg["point_patch_token"]

    def _embed_fusion():
        embeds = inner.embed_tokens(input_ids)
        cur_ids = input_ids[0]
        cur_emb = embeds[0]
        cur_feat = feat_proj.to(device=cur_emb.device, dtype=cur_emb.dtype)
        num_patches = cur_feat.shape[0]
        if use_start_end:
            start_id = cfg["point_start_token"]
            start_pos = torch.where(cur_ids == start_id)[0]
            cur_new = cur_emb
            for pos in reversed(start_pos):
                cur_new = torch.cat([cur_new[:pos + 1], cur_feat, cur_new[pos + num_patches + 1:]])
        else:
            masked = torch.where(cur_ids == patch_id)[0]
            s = masked[0]
            torch.cat([cur_emb[:s], cur_feat, cur_emb[s + num_patches:]])

    embed_ms = cuda_ms(_embed_fusion)
    print(f"  prepare_inputs_embeds (text embed + fusion):      {embed_ms:7.2f} ms")

    # 11. LLM transformer forward
    inputs_embeds = inner.embed_tokens(input_ids)
    dummy_ids = torch.zeros_like(input_ids)

    def _llm():
        hf_prefill(model, dummy_ids, attention_mask,
                   point_clouds=None, inputs_embeds=inputs_embeds)

    llm_ms = cuda_ms(_llm)
    print(f"  LLM forward (32 Llama layers):                    {llm_ms:7.2f} ms")

    total_est = bb_ms + proj_ms + embed_ms + llm_ms
    print(f"  ─── Estimated prefill total:                      {total_est:7.2f} ms")
    print()
    return {
        "fps_ms": fps_ms, "knn_ms": knn_ms, "group_total_ms": group_ms,
        "local_encoder_ms": enc_ms, "reduce_pos_ms": reduce_ms,
        "transformer_ms": trans_ms, "norm_pool_ms": pool_ms,
        "backbone_total_ms": bb_ms, "point_proj_ms": proj_ms,
        "embed_fusion_ms": embed_ms, "llm_forward_ms": llm_ms,
    }


@torch.inference_mode()
def breakdown_batch(model, tok, B_list):
    """Serial-loop vs hypothetical batched breakdown at different B."""
    inner = model.get_model()
    backbone = inner.point_backbone

    input_ids, attention_mask = build_input_ids(tok, model)
    input_ids_b = input_ids.expand(8, -1)
    attention_mask_b = attention_mask.expand(8, -1)

    pcs = [torch.randn(8192, 6, device=DEVICE, dtype=DTYPE) for _ in range(8)]

    print("=== Batch breakdown: point cloud encoding (serial loop vs batched input) ===")
    print(f"{'B':>4}  {'serial_loop_ms':>16}  {'batched_input_ms':>17}  {'speedup':>8}")

    for B in B_list:
        pc_list = pcs[:B]
        pc_batch = torch.stack(pc_list)  # [B, 8192, 6]

        # Serial loop (current implementation)
        def _serial():
            for pc in pc_list:
                backbone(pc.unsqueeze(0))

        serial_ms = cuda_ms(_serial)

        # Batched (if we passed all B point clouds at once)
        def _batched():
            backbone(pc_batch)

        batch_ms = cuda_ms(_batched)
        speedup = serial_ms / batch_ms if batch_ms > 0 else 0
        print(f"  B={B:<2}  serial={serial_ms:8.1f} ms  batched={batch_ms:8.1f} ms  {speedup:6.2f}×")

    print()

    print("=== Batch breakdown: LLM forward at different B ===")
    print(f"{'B':>4}  {'llm_fwd_ms':>12}  {'per_seq_ms':>12}")
    for B in B_list:
        ids_b = input_ids.expand(B, -1)
        mask_b = attention_mask.expand(B, -1)
        dummy = torch.zeros_like(ids_b)
        embeds = inner.embed_tokens(ids_b)

        def _llm_b():
            hf_prefill(model, dummy, mask_b, point_clouds=None, inputs_embeds=embeds)

        llm_ms = cuda_ms(_llm_b)
        print(f"  B={B:<2}  llm={llm_ms:8.1f} ms  per_seq={llm_ms/B:8.1f} ms")

    print()


# ── main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Loading model...")
    model, tok = load_model()
    input_ids, attention_mask = build_input_ids(tok, model)
    print("Model loaded.\n")

    # Single-sequence detailed breakdown
    result = breakdown_single(model, input_ids, attention_mask)

    # Batch comparison
    breakdown_batch(model, tok, B_LIST)

    # Summary: where does time go for B=1?
    bb = result["backbone_total_ms"]
    proj = result["point_proj_ms"]
    embed = result["embed_fusion_ms"]
    llm = result["llm_forward_ms"]
    total = bb + proj + embed + llm
    print("=== Time distribution (B=1 prefill) ===")
    print(f"  PointBERT backbone:    {bb:7.1f} ms  ({100*bb/total:5.1f}%)")
    print(f"    └ FPS:               {result['fps_ms']:7.1f} ms  ({100*result['fps_ms']/total:5.1f}%)")
    print(f"    └ KNN:               {result['knn_ms']:7.1f} ms  ({100*result['knn_ms']/total:5.1f}%)")
    print(f"    └ Local Encoder:     {result['local_encoder_ms']:7.1f} ms  ({100*result['local_encoder_ms']/total:5.1f}%)")
    print(f"    └ Transformer:       {result['transformer_ms']:7.1f} ms  ({100*result['transformer_ms']/total:5.1f}%)")
    print(f"  point_proj:            {proj:7.1f} ms  ({100*proj/total:5.1f}%)")
    print(f"  embed+fusion:          {embed:7.1f} ms  ({100*embed/total:5.1f}%)")
    print(f"  LLM forward (32L):     {llm:7.1f} ms  ({100*llm/total:5.1f}%)")
    print(f"  ─────────────────────────────────────")
    print(f"  Total:                 {total:7.1f} ms  (100.0%)")
