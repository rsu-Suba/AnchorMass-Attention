import os
import sys
import json
import math
import time
import yaml
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import sentencepiece as spm

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from anchormass.standard_model import GPTLanguageModel, GPTConfig
from anchormass.model import AnchorMassForCausalLM, AnchorMassConfig
from eval.utils import expand_standard_rope


def load_standard(device, ckpt_path):
    print(f"[Loading Standard-150M from {ckpt_path}]")
    with open(ROOT_DIR / "anchormass/config.yaml") as f:
        b_cfg = yaml.safe_load(f)["standard_150M"]
    gpt_cfg = GPTConfig(
        vocab_size=b_cfg["VOCAB_SIZE"], dim=b_cfg["DIM"], ff_dim=b_cfg.get("FF_DIM", b_cfg["DIM"] * 4),
        num_heads=b_cfg["NUM_HEADS"], num_kv_heads=b_cfg["NUM_KV_HEADS"], num_layers=b_cfg["NUM_LAYERS"],
        seq_len=b_cfg.get("SEQ_LEN", 1024), dropout=0.0
    )
    model = GPTLanguageModel(gpt_cfg).to(device=device, dtype=torch.bfloat16).eval()
    sd = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    sd = {k.replace("._orig_mod.", ".").removeprefix("_orig_mod."): v for k, v in sd.items()}
    sd = {k: v for k, v in sd.items() if not any(x in k for x in ("cos_cached", "sin_cached", "inv_freq"))}
    model.load_state_dict(sd, strict=False)
    expand_standard_rope(model, max_len=4096)
    return model


def load_anchormass(device, ckpt_path):
    print(f"[Loading AnchorMass-150M from {ckpt_path}]")
    with open(ROOT_DIR / "anchormass/config.yaml") as f:
        l_cfg = yaml.safe_load(f)["anchormass_150M"]
    l_cfg["MAX_GLOBAL_CACHE"] = 8192
    l_cfg["MAX_POSITION_EMBEDDINGS"] = 262144
    l_cfg["ROPE_SCALING_TYPE"] = "dynamic_ntk"
    l_cfg["EVICTION_POLICY"] = "adaptive"
    cfg = AnchorMassConfig.from_dict(l_cfg)
    model = AnchorMassForCausalLM(cfg).to(device=device, dtype=torch.bfloat16).eval()
    sd = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    sd = {k.replace("._orig_mod.", ".").removeprefix("_orig_mod."): v for k, v in sd.items()}
    sd = {k: v for k, v in sd.items() if not any(x in k for x in ("cos_cached", "sin_cached", "inv_freq"))}
    model.load_state_dict(sd, strict=False)
    return model


@torch.no_grad()
def compute_choice_log_likelihood(model, sp, prefix_text, choice_text, device, is_anchormass=False):
    p_ids = sp.encode(prefix_text, out_type=int)
    c_ids = sp.encode(choice_text, out_type=int)
    full_ids = p_ids + c_ids

    x = torch.tensor([full_ids[:-1]], dtype=torch.long, device=device)
    y = torch.tensor([full_ids[1:]], dtype=torch.long, device=device)

    c_len = len(c_ids)
    if c_len == 0:
        return -float("inf")

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        if is_anchormass:
            model.set_session_length(len(full_ids), device=device)
            h = model.embed(x)
            for layer in model.layers:
                layer_fn = getattr(layer, "_orig_mod", layer)
                h, _ = layer_fn(h, kv_cache=None, offset=0, use_cache=False)
            logits = model.output_proj(model.final_norm(h))
        else:
            model.set_session_length(len(full_ids), device=device)
            logits = model(x)

        choice_logits = logits[0, -c_len:, :]
        choice_targets = y[0, -c_len:]

        log_probs = F.log_softmax(choice_logits, dim=-1)
        target_log_probs = log_probs.gather(1, choice_targets.unsqueeze(1)).squeeze(1)
        total_ll = target_log_probs.sum().item()

    norm_ll = total_ll / c_len
    return norm_ll


def evaluate_jcommonsenseqa(model, sp, items, device, is_anchormass=False):
    correct = 0
    total = len(items)
    t0 = time.time()

    for i, item in enumerate(items):
        q = item["question"]
        label = item["label"]
        prefix = f"質問: {q}\n答え: "

        scores = []
        for c_idx in range(5):
            choice = item[f"choice{c_idx}"]
            ll = compute_choice_log_likelihood(model, sp, prefix, choice, device, is_anchormass=is_anchormass)
            scores.append(ll)

        pred = int(np.argmax(scores))
        if pred == label:
            correct += 1

        if (i + 1) % 20 == 0 or (i + 1) == total:
            acc = correct / (i + 1) * 100
            elapsed = time.time() - t0
            print(f"  [{'AnchorMass' if is_anchormass else 'Standard'}] {i+1:4d}/{total} | Acc: {acc:5.2f}% ({correct}/{i+1}) | Speed: {elapsed/(i+1):.3f}s/q")

    final_acc = (correct / total) * 100
    return final_acc, correct, total


def main():
    parser = argparse.ArgumentParser(description="JCommonsenseQA Benchmark")
    parser.add_argument("--std_ckpt", type=str, default="checkpoints/standard/tf_1_150M_4k_warmup.pt")
    parser.add_argument("--am_ckpt", type=str, default="checkpoints/anchormass/lb_11_150M_4k_warmup.pt")
    parser.add_argument("--data_file", type=str, default="data/benchmarks/jcommonsenseqa_val.json")
    parser.add_argument("--limit", type=int, default=200, help="Number of questions (0 = all 1119)")
    parser.add_argument("--report_out", type=str, default="logs/evaluation_jcommonsenseqa_summary.md")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sp_path = ROOT_DIR / "tokenizer/tokenizer.model"
    sp = spm.SentencePieceProcessor(model_file=str(sp_path))

    data_path = ROOT_DIR / args.data_file
    with open(data_path, "r", encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()]

    if args.limit > 0:
        items = items[:args.limit]

    std_path = ROOT_DIR / args.std_ckpt
    if not std_path.exists() and "checkpoints/standard/" in str(std_path):
        fallback = ROOT_DIR / str(args.std_ckpt).replace("checkpoints/standard/", "checkpoints/baseline/")
        if fallback.exists():
            std_path = fallback

    print(f"JCommonsenseQA Benchmark ({len(items)} questions)")
    print(f"Standard Checkpoint  : {std_path}")
    print(f"AnchorMass Checkpoint: {args.am_ckpt}")

    print("\n--- Evaluating Standard 150M ---")
    std_model = load_standard(device, str(std_path))
    s_acc, s_corr, s_tot = evaluate_jcommonsenseqa(std_model, sp, items, device, is_anchormass=False)
    del std_model
    torch.cuda.empty_cache()

    print("\n--- Evaluating AnchorMass 150M ---")
    am_model = load_anchormass(device, str(ROOT_DIR / args.am_ckpt))
    a_acc, a_corr, a_tot = evaluate_jcommonsenseqa(am_model, sp, items, device, is_anchormass=True)
    del am_model
    torch.cuda.empty_cache()

    diff = a_acc - s_acc

    print(f"Final JCommonsenseQA Result ({s_tot} Questions)")
    print(f"Standard 150M   Accuracy: {s_acc:5.2f}% ({s_corr}/{s_tot})")
    print(f"AnchorMass 150M Accuracy: {a_acc:5.2f}% ({a_corr}/{a_tot})")
    print(f"Difference (AM - Std)   : {diff:+5.2f}%")

    md = []
    md.append("# JCommonsenseQA 正規化客観ベンチマーク 評価成果報告書\n")
    md.append(f"- **実施日時**: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    md.append(f"- **ハードウェア**: {torch.cuda.get_device_name(0)}")
    md.append(f"- **評価問題数**: {s_tot} 問 (JGLUE v1.3 JCommonsenseQA Validation)")
    md.append(f"- **評価手法**: 条件付き対数尤度（Normalized Log-Likelihood）5択選択 (LM-Eval-Harness標準プロトコル)")
    md.append(f"- **Standard モデル**: `{args.std_ckpt}`")
    md.append(f"- **AnchorMass モデル**: `{args.am_ckpt}`\n")

    md.append("## 1. 常識推論 正解率 (Accuracy) 比較\n")
    md.append("| モデル | 正解数 / 問題数 | 正解率 (Accuracy) | 判定 |")
    md.append("| :--- | :---: | :---: | :---: |")
    md.append(f"| **Standard-150M** | {s_corr} / {s_tot} | **{s_acc:.2f}%** | 基準 |")
    md.append(f"| **AnchorMass-150M** | {a_corr} / {a_tot} | **{a_acc:.2f}%** | {diff:+.2f}% ({'精度同等' if abs(diff) <= 2.0 else ('AnchorMass 優位' if diff > 0 else 'Standard 優位')}) |\n")


    out_p = ROOT_DIR / args.report_out
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")

    print(f"Report saved to: {out_p}")


if __name__ == "__main__":
    main()
