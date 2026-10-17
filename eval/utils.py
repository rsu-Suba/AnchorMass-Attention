import os
import sys
import math
import time
import yaml
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import sentencepiece as spm
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from anchormass.standard_model import GPTLanguageModel, GPTConfig, rotate_half
from anchormass.model import AnchorMassForCausalLM, AnchorMassConfig

REPORT_PATH = ROOT_DIR / "experiments/long_context_accuracy_report.md"

def expand_standard_rope(model, max_len=300000):
    for block in model.blocks:
        rope = block.attn.rope
        head_dim = rope.cos_cached.shape[-1]
        device = rope.cos_cached.device
        with torch.amp.autocast(device_type="cuda", enabled=False):
            inv_freq = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
            t = torch.arange(max_len, device=device, dtype=torch.float32)
            freqs = torch.outer(t, inv_freq)
            emb = torch.cat([freqs, freqs], dim=-1)
            rope.cos_cached = emb.cos()
            rope.sin_cached = emb.sin()


def load_models(device, dtype, use_sft=True):
    print(f"\n[Loading Models (Mode: {'SFT' if use_sft else 'Base'})...]")
    with open(ROOT_DIR / "anchormass/config.yaml") as f:
        full_cfg = yaml.safe_load(f)

    std_cfg = full_cfg["standard_150M"]
    gpt_cfg = GPTConfig(
        vocab_size=std_cfg["VOCAB_SIZE"], dim=std_cfg["DIM"], ff_dim=std_cfg.get("FF_DIM", std_cfg["DIM"] * 4),
        num_heads=std_cfg["NUM_HEADS"], num_kv_heads=std_cfg["NUM_KV_HEADS"], num_layers=std_cfg["NUM_LAYERS"],
        seq_len=std_cfg["SEQ_LEN"], dropout=0.0
    )
    standard_model = GPTLanguageModel(gpt_cfg).to(device=device, dtype=dtype).eval()
    s_names = ["standard_150M_32k_sft.pt", "tf_1_150M_32k_sft.pt"] if use_sft else ["standard_150M.pt", "tf_1_150M.pt"]
    s_ckpt = None
    for s_name in s_names:
        for d in ["checkpoints/standard/v2", "checkpoints/baseline/v2", "checkpoints/standard", "checkpoints/baseline"]:
            p = ROOT_DIR / d / s_name
            if p.exists():
                s_ckpt = p
                break
        if s_ckpt is not None:
            break
    if s_ckpt is None:
        s_ckpt = ROOT_DIR / "checkpoints/standard/v2" / s_names[0]

    if s_ckpt.exists():
        sd = torch.load(s_ckpt, map_location=device, weights_only=False)
        if isinstance(sd, dict) and "model_state_dict" in sd:
            sd = sd["model_state_dict"]
        sd = {k.replace("._orig_mod.", ".").removeprefix("_orig_mod."): v for k, v in sd.items()}
        standard_model.load_state_dict(sd, strict=False)
    expand_standard_rope(standard_model, max_len=300000)
    print(f"  Loaded Standard-150M ({s_ckpt.name})")

    am_cfg_dict = full_cfg["anchormass_150M"]
    am_cfg = AnchorMassConfig.from_dict(am_cfg_dict)
    anchormass_model = AnchorMassForCausalLM(am_cfg).to(device=device, dtype=dtype).eval()
    am_names = ["anchormass_150M_32k_sft.pt", "lb_11_150M_32k_sft.pt"] if use_sft else ["anchormass_150M.pt", "lb_11_150M.pt"]
    am_ckpt = None
    for am_name in am_names:
        for d in ["checkpoints/anchormass/v2", "checkpoints/anchormass"]:
            p = ROOT_DIR / d / am_name
            if p.exists():
                am_ckpt = p
                break
        if am_ckpt is not None:
            break
    if am_ckpt is None:
        am_ckpt = ROOT_DIR / "checkpoints/anchormass/v2" / am_names[0]

    if am_ckpt.exists():
        sd = torch.load(am_ckpt, map_location=device, weights_only=False)
        if isinstance(sd, dict) and "model_state_dict" in sd:
            sd = sd["model_state_dict"]
        sd = {k.replace("._orig_mod.", ".").removeprefix("_orig_mod."): v for k, v in sd.items()}
        anchormass_model.load_state_dict(sd, strict=False)
    print(f"  Loaded AnchorMass-150M ({am_ckpt.name})")
    return standard_model, anchormass_model

def get_haystack_tokens(target_len, sp, rng=None, offset=None):
    val_bin = ROOT_DIR / "data/corpus/token/val.bin"
    if val_bin.exists():
        data = np.memmap(str(val_bin), dtype=np.uint16, mode='r')
        if len(data) >= target_len:
            max_offset = len(data) - target_len
            if offset is not None:
                start = offset % max(1, max_offset + 1)
            elif rng is not None:
                start = rng.randint(0, max_offset)
            else:
                start = 0
            return data[start : start + target_len].astype(int).tolist()
        else:
            rep = math.ceil(target_len / len(data))
            return (data.astype(int).tolist() * rep)[:target_len]
    sample_text = "自然言語処理と深層学習の技術は急速に進化しており、長大なコンテキストを保持しつつ推論を行うアーキテクチャの開発が世界中で活発に行われています。"
    ids = sp.encode(sample_text, out_type=int)
    rep = math.ceil(target_len / len(ids))
    return (ids * rep)[:target_len]

@torch.no_grad()
def prefill_standard_zero_oom(model, prompt_ids, chunk_size=1024, return_last_chunk_logits=False):
    B, P = prompt_ids.shape
    device = prompt_ids.device
    model.set_session_length(P, device=device)
    num_layers = len(model.blocks)
    num_kv_heads = model.blocks[0].attn.num_kv_heads
    num_heads = model.blocks[0].attn.num_heads
    head_dim = model.blocks[0].attn.head_dim
    dim = model.cfg.dim
    dtype = model.blocks[0].attn.q_proj.weight.dtype

    k_caches = [torch.zeros(B, num_kv_heads, P, head_dim, device=device, dtype=dtype) for _ in range(num_layers)]
    v_caches = [torch.zeros(B, num_kv_heads, P, head_dim, device=device, dtype=dtype) for _ in range(num_layers)]

    total_chunks = math.ceil(P / chunk_size)
    for c_idx, start in enumerate(range(0, P, chunk_size)):
        end = min(start + chunk_size, P)
        chunk = prompt_ids[:, start:end]
        c_len = chunk.shape[1]

        h = model.embed(chunk)
        seq_pos = torch.arange(start, end, device=device)

        for i, block in enumerate(model.blocks):
            h_norm = block.norm1(h)
            q = block.attn.q_proj(h_norm).view(B, c_len, num_heads, head_dim)
            k = block.attn.k_proj(h_norm).view(B, c_len, num_kv_heads, head_dim)
            v = block.attn.v_proj(h_norm).view(B, c_len, num_kv_heads, head_dim)

            cos = block.attn.rope.cos_cached[seq_pos].unsqueeze(0).unsqueeze(2).to(dtype=q.dtype)
            sin = block.attn.rope.sin_cached[seq_pos].unsqueeze(0).unsqueeze(2).to(dtype=q.dtype)
            q = (q * cos) + (rotate_half(q) * sin)
            k = (k * cos) + (rotate_half(k) * sin)

            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

            k_caches[i][:, :, start:end, :] = k
            v_caches[i][:, :, start:end, :] = v

            k_past = k_caches[i][:, :, :end, :]
            v_past = v_caches[i][:, :, :end, :]

            k_pos = torch.arange(end, device=device).unsqueeze(0)
            q_pos = seq_pos.unsqueeze(1)
            causal_mask = (k_pos <= q_pos).unsqueeze(0).unsqueeze(0)
            rep = num_heads // num_kv_heads
            if rep > 1:
                k_past = k_past.repeat_interleave(rep, dim=1)
                v_past = v_past.repeat_interleave(rep, dim=1)
            out = F.scaled_dot_product_attention(
                q, k_past, v_past,
                attn_mask=causal_mask
            )
            out = out.transpose(1, 2).contiguous().view(B, c_len, dim)
            attn_out = block.attn.out_proj(out)
            h = h + attn_out
            h = h + block.ffn(block.norm2(h))

        if total_chunks > 4 and (c_idx + 1) % max(1, total_chunks // 8) == 0:
            pct = (c_idx + 1) / total_chunks * 100
            print(f"    Standard: [ {end//1024}K / {P//1024}K ] ({pct:5.1f}%) | VRAM: {torch.cuda.max_memory_allocated() / 1024**3:4.1f} GB")

    if return_last_chunk_logits:
        logits = model.lm_head(model.norm(h))
    else:
        logits = model.lm_head(model.norm(h)[:, -1, :])
    caches = list(zip(k_caches, v_caches))
    return caches, logits


@torch.no_grad()
def prefill_anchormass(model, prompt_ids, chunk_size=1024, return_last_chunk_logits=False):
    model.eval()
    B, P = prompt_ids.shape
    c_size = min(chunk_size, 1024)
    if not return_last_chunk_logits:
        caches, logits = model.prefill_chunked(prompt_ids, chunk_size=c_size)
        return caches, logits

    model.set_session_length(P, device=prompt_ids.device)
    target_len = min(P, 1024)
    prefix_len = P - target_len

    if prefix_len > 0:
        prefix_ids = prompt_ids[:, :prefix_len]
        caches, _ = model.prefill_chunked(prefix_ids, chunk_size=c_size, total_session_len=P)
        target_ids = prompt_ids[:, prefix_len:]
        h = model.embed(target_ids)
        for i, layer in enumerate(model.layers):
            layer_fn = getattr(layer, "_orig_mod", layer)
            h, caches[i] = layer_fn(h, kv_cache=caches[i], offset=prefix_len, use_cache=True)
    else:
        caches = [None] * len(model.layers)
        h = model.embed(prompt_ids)
        for i, layer in enumerate(model.layers):
            layer_fn = getattr(layer, "_orig_mod", layer)
            h, caches[i] = layer_fn(h, kv_cache=None, offset=0, use_cache=True)

    logits = model.output_proj(model.final_norm(h))
    return caches, logits


@torch.no_grad()
def generate_tokens(model, caches, cur_pos, first_logits, max_tokens=25, is_anchormass=False, repetition_penalty=1.0, eos_ids=(2, 3)):
    gen_ids = []
    logits = first_logits.clone()
    for step in range(max_tokens):
        if repetition_penalty != 1.0 and len(gen_ids) > 0:
            for prev_tok in set(gen_ids):
                if logits.dim() == 2:
                    if logits[0, prev_tok] > 0:
                        logits[0, prev_tok] /= repetition_penalty
                    else:
                        logits[0, prev_tok] *= repetition_penalty
                elif logits.dim() == 3:
                    if logits[0, -1, prev_tok] > 0:
                        logits[0, -1, prev_tok] /= repetition_penalty
                    else:
                        logits[0, -1, prev_tok] *= repetition_penalty

        next_tok = torch.argmax(logits, dim=-1, keepdim=True)
        if next_tok.dim() == 3:
            next_tok = next_tok[:, -1:, :]
        elif next_tok.dim() == 1:
            next_tok = next_tok.unsqueeze(0).unsqueeze(0)
        elif next_tok.dim() == 2 and next_tok.shape[1] > 1:
            next_tok = next_tok[:, -1:]

        tok_id = next_tok.squeeze().item() if next_tok.numel() == 1 else next_tok[0, -1].item()

        if tok_id in eos_ids:
            break

        gen_ids.append(tok_id)

        h = model.embed(next_tok)
        if is_anchormass:
            for i, layer in enumerate(model.layers):
                layer_fn = getattr(layer, "_orig_mod", layer)
                h, caches[i] = layer_fn(h, kv_cache=caches[i], offset=cur_pos, use_cache=True)
            logits = model.output_proj(model.final_norm(h)[:, -1, :])
        else:
            for i, block in enumerate(model.blocks):
                h, caches[i] = block(h, kv_cache=caches[i], offset=cur_pos)
            logits = model.lm_head(model.norm(h)[:, -1, :])
        cur_pos += 1
    return gen_ids

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", action="store_true", help="Use base model instead of SFT")
    args = parser.parse_args()
    use_sft = not args.base

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)

    print("\n" + "=" * 80)
    print(f"🔬 LONG-CONTEXT ACCURACY BENCHMARK: NEEDLE-IN-A-HAYSTACK (4K / 32K / 256K)")
    print(f"Device: {gpu_name} ({vram_gb:.1f} GB VRAM) | Model Mode: {'SFT' if use_sft else 'Base'}")
    print("=" * 80)

    sp_path = ROOT_DIR / "tokenizer/tokenizer.model"
    sp = spm.SentencePieceProcessor(model_file=str(sp_path))

    standard_model, anchormass_model = load_models(device, dtype, use_sft=use_sft)

    test_lengths = [4096, 32768, 262144]
    passkey = "8492"

    if use_sft:
        needle_text = f"\nユーザー: 重要機密事項です。今回の秘密パスキーは【{passkey}】です。忘れないでください。\nアシスタント: 了解いたしました。秘密パスキー【{passkey}】を記憶しました。\n"
        query_text = "ユーザー: 先ほど記録した秘密パスキーは何ですか？パスキーの数字のみ答えてください。\nアシスタント: 秘密パスキーは【"
    else:
        needle_text = f"\n【重要連絡: 今回の秘密パスキーは {passkey} です。】\n"
        query_text = "\n質問: 今回の秘密パスキーは何ですか？ 回答: 秘密パスキーは"

    needle_ids = sp.encode(needle_text, out_type=int)
    query_ids = sp.encode(query_text, out_type=int)

    results = []

    print("\n" + "-" * 85)
    print(f"{'Context Length':>15} | {'Standard 判定':>13} | {'AnchorMass 判定':>13} | {'Std Prefill':>13} | {'AM Prefill':>13}")
    print("-" * 85)

    for ctx_len in test_lengths:
        ctx_label = f"{ctx_len // 1024}K ({ctx_len:,})"

        haystack = get_haystack_tokens(ctx_len - len(needle_ids) - len(query_ids), sp)
        mid_idx = len(haystack) // 2
        full_prompt_ids = haystack[:mid_idx] + needle_ids + haystack[mid_idx:] + query_ids
        prompt_tensor = torch.tensor([full_prompt_ids], dtype=torch.long, device=device)
        P = prompt_tensor.shape[1]

        torch.cuda.empty_cache()
        t0 = time.perf_counter()
        b_caches, b_logits = prefill_standard_zero_oom(standard_model, prompt_tensor, chunk_size=1024)
        b_prefill_sec = time.perf_counter() - t0

        b_gen_ids = generate_tokens(standard_model, b_caches, P, b_logits, max_tokens=15, is_anchormass=False)
        b_gen_text = sp.decode(b_gen_ids).strip()
        b_pass = passkey in b_gen_text
        del b_caches, b_logits
        torch.cuda.empty_cache()

        t0 = time.perf_counter()
        l_caches, l_logits = prefill_anchormass(anchormass_model, prompt_tensor)
        l_prefill_sec = time.perf_counter() - t0

        l_gen_ids = generate_tokens(anchormass_model, l_caches, P, l_logits, max_tokens=15, is_anchormass=True)
        l_gen_text = sp.decode(l_gen_ids).strip()
        l_pass = passkey in l_gen_text
        del l_caches, l_logits
        torch.cuda.empty_cache()

        b_status = "⭕ 正解 (PASS)" if b_pass else "❌ 不正解"
        l_status = "⭕ 正解 (PASS)" if l_pass else "❌ 不正解"

        b_time_str = f"{b_prefill_sec:.2f}s"
        l_time_str = f"{l_prefill_sec:.2f}s"
        print(f"{ctx_label:>15} | {b_status:>13} | {l_status:>13} | {b_time_str:>13} | {l_time_str:>13}")
        print(f"   └─ Standard 出力: \"{b_gen_text[:40]}\"")
        print(f"   └─ AnchorMass 出力: \"{l_gen_text[:40]}\"")

        results.append({
            "context": ctx_len,
            "ctx_label": ctx_label,
            "b_pass": b_pass, "b_text": b_gen_text, "b_prefill": b_prefill_sec,
            "l_pass": l_pass, "l_text": l_gen_text, "l_prefill": l_prefill_sec,
        })

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("#超長文ベンチマーク（Needle-in-a-Haystack 4K / 32K / 256K）実証レポート\n\n")
        f.write(f"- **ハードウェア**: {gpu_name} ({vram_gb:.1f} GB VRAM)\n")
        f.write(f"- **比較対象**: Standard-150M vs AnchorMass-150M [{ 'SFT' if use_sft else 'Base' }]\n")
        f.write(f"- **ターゲットパスキー**: `{passkey}` (コンテキスト 50% の深さに埋め込み)\n\n")

        f.write("## 1. Needle-in-a-Haystack 正解判定一覧表\n\n")
        f.write("| コンテキスト長 | Standard 判定 | Standard 出力 | **AnchorMass 判定** | **AnchorMass 出力** | Prefill 時間比較 |\n")
        f.write("| :--- | :---: | :--- | :---: | :--- | :---: |\n")

        for r in results:
            b_s = "正解" if r["b_pass"] else "不正解"
            l_s = "正解" if r["l_pass"] else "不正解"
            f.write(f"| **{r['ctx_label']}** | {b_s} | `{r['b_text']}` | {l_s} | `{r['l_text']}` | Std: {r['b_prefill']:.2f}s / AM: {r['l_prefill']:.2f}s |\n")


    print(f"\n[DONE] Saved accuracy report to: {REPORT_PATH}")

if __name__ == "__main__":
    main()
