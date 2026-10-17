import os
import sys
import math
import time
import json
import yaml
import random
import statistics
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
import sentencepiece as spm

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from anchormass.standard_model import GPTLanguageModel, GPTConfig, rotate_half
from anchormass.model import AnchorMassForCausalLM, AnchorMassConfig
from eval.utils import prefill_standard_zero_oom, prefill_anchormass, generate_tokens, expand_standard_rope, get_haystack_tokens

TEMPLATES = {
    "original_prefix": (
        "\nユーザー: 重要機密事項です。今回の秘密パスキーは【{k}】です。忘れないでください。\n"
        "アシスタント: 了解いたしました。秘密パスキー【{k}】を記憶しました。\n",
        "ユーザー: 先ほど記録した秘密パスキーは何ですか？パスキーの数字のみ答えてください。\n"
        "アシスタント: 秘密パスキーは【",
    ),
    "original_no_prefix": (
        "\nユーザー: 重要機密事項です。今回の秘密パスキーは【{k}】です。忘れないでください。\n"
        "アシスタント: 了解いたしました。秘密パスキー【{k}】を記憶しました。\n",
        "ユーザー: 先ほど記録した秘密パスキーは何ですか？余計な解説は一切含めず、4桁の半角数字のみで答えてください。\n"
        "アシスタント:",
    ),
    "lexical_paraphrase": (
        "\nユーザー: 忘れないでください。秘密の暗証番号は【{k}】です。\n"
        "アシスタント: わかりました。暗証番号【{k}】を覚えておきます。\n",
        "ユーザー: 先ほどの暗証番号を教えてください。数字だけで答えてください。\n"
        "アシスタント:",
    ),
    "memo_format": (
        "\nメモ: 私の秘密コードは「{k}」です。\n",
        "質問: メモに書かれていた秘密コードは何ですか？数字だけ答えてください。\n回答:",
    ),
    "dialogue_format": (
        "\n田中: 会議室の解錠コードは【{k}】にしておいたよ。\n佐藤: 了解です！\n",
        "田中: 会議室の解錠コードは何番？数字だけ教えて。\n佐藤:",
    ),
}


def load_standard_model(device, dtype, ckpt_path):
    print(f"[Loading Standard-150M from {ckpt_path}]")
    with open(ROOT_DIR / "anchormass/config.yaml") as f:
        b_cfg = yaml.safe_load(f)["standard_150M"]
    gpt_cfg = GPTConfig(
        vocab_size=b_cfg["VOCAB_SIZE"], dim=b_cfg["DIM"], ff_dim=b_cfg.get("FF_DIM", b_cfg["DIM"] * 4),
        num_heads=b_cfg["NUM_HEADS"], num_kv_heads=b_cfg["NUM_KV_HEADS"], num_layers=b_cfg["NUM_LAYERS"],
        seq_len=b_cfg.get("SEQ_LEN", 1024), dropout=0.0
    )
    model = GPTLanguageModel(gpt_cfg).to(device=device, dtype=dtype).eval()
    if os.path.exists(ckpt_path):
        sd = torch.load(ckpt_path, map_location=device, weights_only=False)
        if isinstance(sd, dict) and "model_state_dict" in sd:
            sd = sd["model_state_dict"]
        sd = {k.replace("._orig_mod.", ".").removeprefix("_orig_mod."): v for k, v in sd.items()}
        sd = {k: v for k, v in sd.items() if not any(x in k for x in ("cos_cached", "sin_cached", "inv_freq"))}
        model.load_state_dict(sd, strict=False)
        print("  Standard weights loaded successfully.")
    else:
        print(f"  WARNING: Checkpoint {ckpt_path} not found! Using random weights.")
    expand_standard_rope(model, max_len=300000)
    return model, b_cfg


def load_anchormass_model(device, dtype, ckpt_path):
    print(f"[Loading AnchorMass-150M from {ckpt_path}]")
    with open(ROOT_DIR / "anchormass/config.yaml") as f:
        l_cfg = yaml.safe_load(f)["anchormass_150M"]
    l_cfg["MAX_GLOBAL_CACHE"] = 8192
    l_cfg["MAX_POSITION_EMBEDDINGS"] = 262144
    l_cfg["ROPE_SCALING_TYPE"] = "dynamic_ntk"
    l_cfg["EVICTION_POLICY"] = "adaptive"
    cfg = AnchorMassConfig.from_dict(l_cfg)
    model = AnchorMassForCausalLM(cfg).to(device=device, dtype=dtype).eval()
    if os.path.exists(ckpt_path):
        sd = torch.load(ckpt_path, map_location=device, weights_only=False)
        if isinstance(sd, dict) and "model_state_dict" in sd:
            sd = sd["model_state_dict"]
        sd = {k.replace("._orig_mod.", ".").removeprefix("_orig_mod."): v for k, v in sd.items()}
        sd = {k: v for k, v in sd.items() if not any(x in k for x in ("cos_cached", "sin_cached", "inv_freq"))}
        model.load_state_dict(sd, strict=False)
        print("  AnchorMass weights loaded successfully.")
    else:
        print(f"  WARNING: Checkpoint {ckpt_path} not found! Using random weights.")
    return model, l_cfg


def set_anchormass_policy(model, policy, cache_budget=8192):
    for layer in model.layers:
        a = getattr(layer, "_orig_mod", layer).attn
        a.eviction_policy = policy
        a.max_global_cache = cache_budget


@torch.no_grad()
def prefill_standard_oneshot(model, prompt_ids, return_last_chunk_logits=False):
    B, P = prompt_ids.shape
    device = prompt_ids.device
    model.set_session_length(P, device=device)
    x = model.embed(prompt_ids)
    caches = [] if not return_last_chunk_logits else None
    for block in model.blocks:
        x, new_cache = block(x, kv_cache=None, offset=0, use_cache=(not return_last_chunk_logits))
        if not return_last_chunk_logits:
            caches.append(new_cache)
        else:
            del new_cache
    if return_last_chunk_logits:
        eval_w = min(P, 1024)
        logits = model.lm_head(model.norm(x[:, -eval_w:, :]))
        return None, logits
    else:
        logits = model.lm_head(model.norm(x[:, -1, :]))
        return caches, logits


@torch.no_grad()
def eval_ppl_slice(model, val_tokens, seq_len, sample_anchors, vocab_size, device, is_anchormass=False):
    total_nll = 0.0
    total_toks = 0

    for anchor in sample_anchors:
        chunk_x = val_tokens[anchor - seq_len : anchor].astype(np.int64)
        chunk_y = val_tokens[anchor - seq_len + 1 : anchor + 1].astype(np.int64)
        x = torch.from_numpy(chunk_x.copy()).long().unsqueeze(0).to(device)
        y = torch.from_numpy(chunk_y.copy()).long().unsqueeze(0).to(device)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            if is_anchormass:
                _, logits = prefill_anchormass(model, x, chunk_size=min(seq_len, 1024), return_last_chunk_logits=True)
            else:
                _, logits = prefill_standard_oneshot(model, x, return_last_chunk_logits=True)

            c_len = logits.shape[1]
            target_slice = y[:, -c_len:].contiguous()
            loss_sum = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                target_slice.reshape(-1),
                reduction="sum",
            )
            total_nll += loss_sum.item()
            total_toks += target_slice.numel()

    avg_loss = total_nll / max(1, total_toks)
    ppl = math.exp(avg_loss) if avg_loss < 20 else float("inf")
    return avg_loss, ppl


def get_cache_nbytes(caches) -> int:
    total = 0
    def visit(obj):
        nonlocal total
        if torch.is_tensor(obj):
            total += obj.numel() * obj.element_size()
        elif isinstance(obj, (tuple, list)):
            for item in obj:
                visit(item)
        elif isinstance(obj, dict):
            for item in obj.values():
                visit(item)
    visit(caches)
    return total


@torch.no_grad()
def eval_single_needle(model, sp, ctx_len, depth, template_name, device, is_anchormass=False, passkey=None, needle_seed=None):
    import re
    if passkey is None:
        passkey = str(random.randint(1000, 9999))
    needle_tmpl, query_tmpl = TEMPLATES[template_name]
    needle_text = needle_tmpl.format(k=passkey)
    query_text = query_tmpl

    needle_ids = sp.encode(needle_text, out_type=int)
    query_ids = sp.encode(query_text, out_type=int)

    haystack_len = ctx_len - len(needle_ids) - len(query_ids)
    if haystack_len <= 0:
        return passkey, "", False, 0.0, 0, 0.0, 0.0, {
            "strict": False, "first_number": False, "contains": False,
            "prefill_s": 0.0, "decode_s": 0.0, "n_gen": 0,
        }

    rng = random.Random(needle_seed) if needle_seed is not None else None
    haystack = get_haystack_tokens(haystack_len, sp, rng=rng)
    insert_idx = int(len(haystack) * depth)
    full_ids = haystack[:insert_idx] + needle_ids + haystack[insert_idx:] + query_ids
    t = torch.tensor([full_ids], device=device)

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        if is_anchormass:
            caches, logits = prefill_anchormass(model, t, chunk_size=1024)
        else:
            caches, logits = prefill_standard_oneshot(model, t, return_last_chunk_logits=False)
    torch.cuda.synchronize()
    t1 = time.time()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        gen_toks = generate_tokens(model, caches, t.shape[1], logits, max_tokens=15, is_anchormass=is_anchormass)
    torch.cuda.synchronize()
    t2 = time.time()
    elapsed = t2 - t0
    prefill_s = t1 - t0
    decode_s = t2 - t1
    n_gen = len(gen_toks)
    peak_vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
    kv_mb = get_cache_nbytes(caches) / (1024 ** 2)
    del caches, logits, t

    out_text = sp.decode(gen_toks).strip()
    nums = re.findall(r"\d+", out_text)
    pred_digits = nums[0] if nums else ""
    strict_match = (out_text == passkey)
    first_number_match = (pred_digits == passkey)
    contains_match = (passkey in out_text)
    hit = bool(first_number_match)
    digit_acc = sum(1 for a, b in zip(pred_digits[:4], passkey) if a == b) if pred_digits else 0
    return passkey, out_text, hit, elapsed, digit_acc, peak_vram_gb, kv_mb, {
        "strict": bool(strict_match),
        "first_number": bool(first_number_match),
        "contains": bool(contains_match),
        "prefill_s": prefill_s,
        "decode_s": decode_s,
        "n_gen": n_gen,
    }


def det_seed(*parts) -> int:
    import zlib
    s = ":".join(str(p) for p in parts).encode("utf-8")
    return zlib.crc32(s) & 0x7FFFFFFF


def main():
    parser = argparse.ArgumentParser(description="150M Full Evaluation Suite")
    parser.add_argument("--std_ckpt", type=str, default="checkpoints/standard/v2/tf_1_150M_32k_sft.pt")
    parser.add_argument("--am_ckpt", type=str, default="checkpoints/anchormass/v2/lb_11_150M_32k_sft.pt")
    parser.add_argument("--report_out", type=str, default="logs/evaluation_150M_summary.md")
    parser.add_argument("--ppl_samples", type=int, default=10)
    parser.add_argument("--needle_trials", type=int, default=3)
    parser.add_argument("--ablation_trials", type=int, default=10)
    parser.add_argument("--template_trials", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_ppl", action="store_true", help="Skip PPL test and run Needle/Ablation only")
    parser.add_argument("--only_template", action="store_true", help="Run only Test 4 (Template Generalization)")
    parser.add_argument("--needle_lens", type=int, nargs="+", default=[4096, 8192, 16384, 32768, 65536])
    parser.add_argument("--max_std_len", type=int, default=32768,
                        help="Standardモデルを動かす最大系列長。8GB GPUでは32768、48GB GPUでは262144まで可")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16
    sp_path = ROOT_DIR / "tokenizer/tokenizer.model"
    sp = spm.SentencePieceProcessor(model_file=str(sp_path))

    val_bin_path = ROOT_DIR / "data/corpus/token/val.bin"
    val_tokens = np.memmap(str(val_bin_path), dtype=np.uint16, mode='r')

    print("Start: 150M eval")
    print(f"Device: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"Validation tokens: {len(val_tokens):,} tokens ({val_bin_path})")

    std_path = ROOT_DIR / args.std_ckpt
    if not std_path.exists() and "checkpoints/standard/" in str(std_path):
        fb = ROOT_DIR / str(args.std_ckpt).replace("checkpoints/standard/", "checkpoints/baseline/")
        if fb.exists():
            std_path = fb
    standard_model, _ = load_standard_model(device, dtype, str(std_path))
    anchormass_model, _ = load_anchormass_model(device, dtype, str(ROOT_DIR / args.am_ckpt))

    md_lines = []
    md_lines.append("# 150M 完全統制対照実験 最終ベンチマーク成果報告書\n")
    md_lines.append(f"- **実施日時**: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    md_lines.append(f"- **ハードウェア**: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB)")
    md_lines.append(f"- **Standard モデル**: `{args.std_ckpt}`")
    md_lines.append(f"- **AnchorMass モデル**: `{args.am_ckpt}`")
    md_lines.append(f"- **評価データ**: 完全未見 `data/corpus/token/val.bin`\n")

    if not args.skip_ppl and not args.only_template:
        print("[Test 1/4] 長系列 PPL 評価 (未見コーパス・同一末尾1,024トークン対照評価)")

        ppl_lens = [1024, 2048, 4096, 8192, 16384, 32768]
        ppl_results = []

        max_ctx = 32768
        valid_range = len(val_tokens) - max_ctx - 10
        sample_anchors = [
            max_ctx + int(i * (valid_range // args.ppl_samples))
            for i in range(args.ppl_samples)
        ]

        md_lines.append("## 1. 長系列 PPL (Perplexity) 完全対照評価\n")
        md_lines.append("| 系列長 (Tokens) | Standard-150M (PPL) | Standard Loss | AnchorMass-150M (PPL) | AnchorMass Loss | 差分 (AM - Std) |")
        md_lines.append("| :--- | :---: | :---: | :---: | :---: | :---: |")

        for L in ppl_lens:
            torch.cuda.empty_cache()
            print(f"  Evaluating Seq Len {L:,} ({args.ppl_samples} samples, target 1,024 tokens each)...")
            b_loss, b_ppl = eval_ppl_slice(standard_model, val_tokens, L, sample_anchors, 28000, device, is_anchormass=False)
            l_loss, l_ppl = eval_ppl_slice(anchormass_model, val_tokens, L, sample_anchors, 28000, device, is_anchormass=True)
            diff = l_ppl - b_ppl
            diff_str = f"{diff:+.2f}" if abs(diff) < 1000 else "N/A"
            row = f"| **{L:,}** | {b_ppl:.2f} | {b_loss:.3f} | **{l_ppl:.2f}** | **{l_loss:.3f}** | {diff_str} |"
            print(f"    -> Standard: PPL {b_ppl:.2f} (Loss {b_loss:.3f}) | AnchorMass: PPL {l_ppl:.2f} (Loss {l_loss:.3f})")
            md_lines.append(row)
            ppl_results.append((L, b_ppl, l_ppl))

    else:
        print("\n[INFO] Skipping PPL evaluation")

    if not args.only_template:
        print("[Test 2/4] Needle 全長スキャン想起テスト (Paired 同一パスキー・ランダム背景対照評価)")

        needle_lens = list(args.needle_lens)
        depths = [0.1, 0.3, 0.5, 0.7, 0.9]

        md_lines.append("## 2. 長文想起評価 (Needle-in-a-Haystack 全長スキャン・Paired 対照)\n")
        md_lines.append("| 系列長 | Standard 完全想起率 | Standard 一致桁 | AnchorMass 完全想起率 | AnchorMass 一致桁 | KV Cache / Peak VRAM (AM vs Std) | Prefill 中央値 (Std vs AM) | Decode (Std vs AM) |")
        md_lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

        set_anchormass_policy(anchormass_model, "adaptive", cache_budget=8192)

        for L in needle_lens:
            torch.cuda.empty_cache()
            print(f"\n  Testing Needle at Length {L:,}...")
            b_hits = 0
            l_hits = 0
            b_digits = []
            l_digits = []
            l_vram_max = 0.0
            b_vram_max = 0.0
            l_kv_mb = 0.0
            b_kv_mb = 0.0
            l_pref, l_dec, b_pref, b_dec = [], [], [], []
            l_ns, b_ns = [], []
            total_evals = len(depths) * args.needle_trials
            run_base = (L <= args.max_std_len)
            len_rng = random.Random(det_seed(args.seed, "needle_scan", L))

            for d in depths:
                for trial in range(args.needle_trials):
                    paired_key = str(len_rng.randint(1000, 9999))
                    trial_seed = det_seed(args.seed, "needle_bg", L, d, trial)

                    p, o, h, t_l, d_l, v_l, kv_l, _m_l = eval_single_needle(
                        anchormass_model, sp, L, d, "original_no_prefix", device, is_anchormass=True, passkey=paired_key, needle_seed=trial_seed
                    )
                    if h: l_hits += 1
                    l_digits.append(d_l)
                    l_vram_max = max(l_vram_max, v_l)
                    l_kv_mb = kv_l
                    l_pref.append(_m_l["prefill_s"])
                    l_dec.append(_m_l["decode_s"])
                    l_ns.append(_m_l["n_gen"])
                    print(f"    [AnchorMass] Len {L//1024:2d}K | Depth {d:.1f} | 目標:【{p}】| 出力: \"{o[:22]}\" -> {'🎯 成功' if h else '❌ 失敗'} ({d_l}/4桁, KV:{kv_l:.0f}MB)", flush=True)

                    if run_base:
                        p, o, h, t_b, d_b, v_b, kv_b, _m_b = eval_single_needle(
                            standard_model, sp, L, d, "original_no_prefix", device, is_anchormass=False, passkey=paired_key, needle_seed=trial_seed
                        )
                        if h: b_hits += 1
                        b_digits.append(d_b)
                        b_vram_max = max(b_vram_max, v_b)
                        b_kv_mb = kv_b
                        b_pref.append(_m_b["prefill_s"])
                        b_dec.append(_m_b["decode_s"])
                        b_ns.append(_m_b["n_gen"])
                        print(f"    [Standard]   Len {L//1024:2d}K | Depth {d:.1f} | 目標:【{p}】| 出力: \"{o[:22]}\" -> {'🎯 成功' if h else '❌ 失敗'} ({d_b}/4桁, KV:{kv_b:.0f}MB)", flush=True)

            l_rate = (l_hits / total_evals) * 100
            l_avg_d = sum(l_digits) / len(l_digits) if l_digits else 0.0
            def _med(xs):
                return statistics.median(xs) if xs else float("nan")
            def _tps(ds, ns):
                ts = [n / d for d, n in zip(ds, ns)] if ds and ns else []
                return statistics.median(ts) if ts else float("nan")
            if run_base:
                b_rate = (b_hits / total_evals) * 100
                b_avg_d = sum(b_digits) / len(b_digits) if b_digits else 0.0
                b_str = f"{b_rate:.1f}% ({b_hits}/{total_evals})"
                b_d_str = f"{b_avg_d:.2f} / 4.0"
                vram_str = f"KV {l_kv_mb:.0f}MB ({l_vram_max:.2f}GB) vs KV {b_kv_mb:.0f}MB ({b_vram_max:.2f}GB)"
                pref_str = f"{_med(b_pref):.2f}s vs {_med(l_pref):.2f}s"
                dec_str = f"{_tps(b_dec, b_ns):.1f} vs {_tps(l_dec, l_ns):.1f} tok/s"
            else:
                b_str = "OOM / 制限超過"
                b_d_str = "-"
                vram_str = f"KV {l_kv_mb:.0f}MB ({l_vram_max:.2f}GB) vs OOM (>8GB)"
                pref_str = f"- vs {_med(l_pref):.2f}s"
                dec_str = f"- vs {_tps(l_dec, l_ns):.1f} tok/s"

            row = f"| **{L:,}** | {b_str} | {b_d_str} | **{l_rate:.1f}%** ({l_hits}/{total_evals}) | **{l_avg_d:.2f} / 4.0** | {vram_str} | {pref_str} | {dec_str} |"
            print(f"  >>> Length {L:,} 完了 -> Standard: {b_str} | AnchorMass: {l_rate:.1f}% ({l_avg_d:.2f}/4桁) | {vram_str} | Prefill {pref_str} | Decode {dec_str}")
            md_lines.append(row)

        print(f"[Test 3/4] 内部キャッシュ管理 5者アブレーション対照実験 (Paired, {args.ablation_trials} trials)")

        policies = ["adaptive", "score_merge", "uniform_merge", "priority", "fifo"]
        test_len = 8192
        test_caches = [128, 256, 512]

        md_lines.append("## 3. 内部キャッシュ管理方式 5者アブレーション (同一重み・同一問題での対照比較)\n")
        md_lines.append(f"- **テスト系列長**: {test_len:,} tokens (極限圧縮ストレステスト)\n")
        md_lines.append("| キャッシュ予算 | Adaptive (新ハイブリッド) | Score Merge | Uniform Merge | Priority (間引き) | FIFO (StreamingLLM) |")
        md_lines.append("| :--- | :---: | :---: | :---: | :---: | :---: |")

        for cb in test_caches:
            row_cols = [f"**{cb} slots** ({test_len // cb}x 圧縮)"]
            print(f"  Testing Cache Budget: {cb} slots...")
            trials = args.ablation_trials
            ab_rng = random.Random(det_seed(args.seed, "ablation", test_len, cb))
            paired_keys = [str(ab_rng.randint(1000, 9999)) for _ in range(trials)]
            paired_seeds = [det_seed(args.seed, "ablation_bg", test_len, cb, tr) for tr in range(trials)]
            for pol in policies:
                set_anchormass_policy(anchormass_model, pol, cache_budget=cb)
                hits = 0
                for pk, s_seed in zip(paired_keys, paired_seeds):
                    p, o, h, _, _, _, _, _ = eval_single_needle(
                        anchormass_model, sp, test_len, 0.5, "original_no_prefix", device, is_anchormass=True, passkey=pk, needle_seed=s_seed
                    )
                    if h: hits += 1
                acc = (hits / trials) * 100
                row_cols.append(f"{acc:.0f}% ({hits}/{trials})")
                print(f"    Policy {pol:14s}: {acc:3.0f}% ({hits}/{trials})")
            md_lines.append("| " + " | ".join(row_cols) + " |")

    print(f"[Test 4/4] テンプレート汎化ロバスト性追試 (5パターン・各 {args.template_trials} 試行 Paired)")

    md_lines.append(f"## 4. プロンプト・テンプレート汎化追試 (8K 長文想起・各 {args.template_trials} 試行 Paired)\n")
    md_lines.append("| テンプレート種別 | Standard-150M 完全想起率 (一致桁) | AnchorMass-150M (Adaptive) 完全想起率 (一致桁) | 判定 |")
    md_lines.append("| :--- | :---: | :---: | :---: |")

    set_anchormass_policy(anchormass_model, "adaptive", cache_budget=8192)

    for tmpl_key in TEMPLATES.keys():
        b_hits = 0
        l_hits = 0
        b_digits = []
        l_digits = []
        trials = args.template_trials
        tmpl_rng = random.Random(det_seed(args.seed, "template", tmpl_key))
        for tr in range(trials):
            pk = str(tmpl_rng.randint(1000, 9999))
            tr_seed = det_seed(args.seed, "template_bg", tmpl_key, tr)
            p, o, h_b, _, d_b, _, _, _ = eval_single_needle(standard_model, sp, 8192, 0.5, tmpl_key, device, is_anchormass=False, passkey=pk, needle_seed=tr_seed)
            if h_b: b_hits += 1
            b_digits.append(d_b)
            p, o, h_l, _, d_l, _, _, _ = eval_single_needle(anchormass_model, sp, 8192, 0.5, tmpl_key, device, is_anchormass=True, passkey=pk, needle_seed=tr_seed)
            if h_l: l_hits += 1
            l_digits.append(d_l)
        b_acc = (b_hits / trials) * 100
        l_acc = (l_hits / trials) * 100
        b_avg_d = sum(b_digits) / len(b_digits) if b_digits else 0.0
        l_avg_d = sum(l_digits) / len(l_digits) if l_digits else 0.0
        if b_hits == 0 and l_hits == 0:
            verdict = "❌ 共同失敗（要対策）"
        elif l_acc >= 80.0 or l_acc >= b_acc - 10.0:
            verdict = "✅ 堅牢"
        else:
            verdict = "⚠️ 改善要"
        row = f"| `{tmpl_key}` | {b_acc:.1f}% ({b_hits}/{trials}, {b_avg_d:.2f}/4.0) | **{l_acc:.1f}%** ({l_hits}/{trials}, **{l_avg_d:.2f}/4.0**) | {verdict} |"
        print(f"  Template {tmpl_key:20s} -> Std: {b_acc:5.1f}% ({b_hits}/{trials}, {b_avg_d:.2f}/4桁) | AM: {l_acc:5.1f}% ({l_hits}/{trials}, {l_avg_d:.2f}/4桁)", flush=True)
        md_lines.append(row)

    md_lines.append("\n---\n*本ベンチマーク結果は、すべて完全未見データ `val.bin` と決定論的乱数シード（Paired 課題）により自動生成されました。*\n")

    out_p = ROOT_DIR / args.report_out
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    print(f"Finish: Report saved: {out_p}")


if __name__ == "__main__":
    main()

