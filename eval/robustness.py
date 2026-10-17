import sys
import json
import math
import random
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import sentencepiece as spm
from scipy.stats import binomtest

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from anchormass.standard_model import GPTLanguageModel, GPTConfig
from anchormass.model import AnchorMassForCausalLM, AnchorMassConfig
from eval.commonsense import load_standard, load_anchormass, compute_choice_log_likelihood
from eval.long_context import (
    get_haystack_tokens,
    prefill_anchormass,
    generate_tokens,
    set_anchormass_policy,
    det_seed,
)


def wilson_score_interval(successes: int, trials: int, confidence: float = 0.95):
    if trials == 0:
        return 0.0, 0.0
    z = 1.95996
    p_hat = successes / trials
    denom = 1.0 + (z**2) / trials
    center = (p_hat + (z**2) / (2.0 * trials)) / denom
    half_width = (z * math.sqrt((p_hat * (1.0 - p_hat) + (z**2) / (4.0 * trials)) / trials)) / denom
    return max(0.0, (center - half_width) * 100.0), min(100.0, (center + half_width) * 100.0)


def paired_mcnemar_exact(b_only: int, l_only: int):
    total = b_only + l_only
    if total == 0:
        return 1.0
    res = binomtest(b_only, n=total, p=0.5, alternative="two-sided")
    return float(res.pvalue)



def eval_stage_jcqa(sp, device, std_ckpt, am_ckpt, dataset, limit=None):
    std_model = load_standard(device, std_ckpt)
    am_model = load_anchormass(device, am_ckpt)

    eval_data = dataset[:limit] if limit else dataset
    std_hits = []
    am_hits = []

    for i, item in enumerate(eval_data):
        q = item["question"]
        choices = [item[f"choice{c}"] for c in range(5)]
        label = item["label"]
        prefix = f"質問: {q}\n答え: "

        std_scores = [
            compute_choice_log_likelihood(std_model, sp, prefix, c, device, is_anchormass=False)
            for c in choices
        ]
        am_scores = [
            compute_choice_log_likelihood(am_model, sp, prefix, c, device, is_anchormass=True)
            for c in choices
        ]

        std_hits.append(int(int(np.argmax(std_scores)) == label))
        am_hits.append(int(int(np.argmax(am_scores)) == label))

    del std_model, am_model
    torch.cuda.empty_cache()

    n = len(eval_data)
    s_corr = sum(std_hits)
    a_corr = sum(am_hits)
    s_lo, s_hi = wilson_score_interval(s_corr, n)
    a_lo, a_hi = wilson_score_interval(a_corr, n)

    both_c = sum(1 for s, a in zip(std_hits, am_hits) if s and a)
    s_only = sum(1 for s, a in zip(std_hits, am_hits) if s and not a)
    a_only = sum(1 for s, a in zip(std_hits, am_hits) if not s and a)
    both_w = sum(1 for s, a in zip(std_hits, am_hits) if not s and not a)

    p_val = paired_mcnemar_exact(s_only, a_only)
    diff = (a_corr - s_corr) / n * 100.0
    p10 = s_only / n
    p01 = a_only / n
    var_diff = ((p10 + p01) - (p01 - p10) ** 2) / n
    diff_se = math.sqrt(max(0.0, var_diff)) * 100.0

    return {
        "n": n,
        "std_correct": s_corr,
        "std_acc": s_corr / n * 100.0,
        "std_ci95": [s_lo, s_hi],
        "anchormass_correct": a_corr,
        "anchormass_acc": a_corr / n * 100.0,
        "anchormass_ci95": [a_lo, a_hi],
        "diff_pct": diff,
        "diff_ci95": [diff - 1.96 * diff_se, diff + 1.96 * diff_se],
        "both_correct": both_c,
        "std_only": s_only,
        "anchormass_only": a_only,
        "both_wrong": both_w,
        "mcnemar_p": p_val,
    }


def eval_ablation_5way_n30(sp, device, am_ckpt, seed=42):
    am_model = load_anchormass(device, am_ckpt)
    test_len = 8192
    budgets = [512, 256, 128]
    policies = ["adaptive", "score_merge", "uniform_merge", "priority", "fifo"]
    depths = [0.1, 0.3, 0.5, 0.7, 0.9]
    trials_per_depth = 6
    total_trials = len(depths) * trials_per_depth

    results = {}

    for budget in budgets:
        results[str(budget)] = {}
        trial_keys = []
        trial_depths = []
        trial_seeds = []
        for d in depths:
            for tr in range(trials_per_depth):
                s = det_seed(seed, "ablation_n30", test_len, budget, int(d * 100), tr)
                rng = random.Random(s)
                trial_keys.append(str(rng.randint(1000, 9999)))
                trial_depths.append(d)
                trial_seeds.append(det_seed(seed, "bg_n30", test_len, budget, int(d * 100), tr))

        policy_hits = {}
        policy_digits = {}
        for pol in policies:
            set_anchormass_policy(am_model, pol, cache_budget=budget)
            hits = []
            digits = []
            for passkey, depth, s_seed in zip(trial_keys, trial_depths, trial_seeds):
                needle = f"\nユーザー: 重要機密事項です。今回の秘密パスキーは【{passkey}】です。忘れないでください。\nアシスタント: 了解いたしました。秘密パスキー【{passkey}】を記憶しました。\n"
                query = "ユーザー: 先ほど記録した秘密パスキーは何ですか？余計な解説は一切含めず、4桁の半角数字のみで答えてください。\nアシスタント: "
                n_ids = sp.encode(needle, out_type=int)
                q_ids = sp.encode(query, out_type=int)
                h_len = test_len - len(n_ids) - len(q_ids)
                haystack = get_haystack_tokens(h_len, sp, seed=s_seed)
                insert_idx = int(len(haystack) * depth)
                full_ids = haystack[:insert_idx] + n_ids + haystack[insert_idx:] + q_ids

                t = torch.tensor([full_ids], device=device)
                caches, logits = prefill_anchormass(am_model, t, chunk_size=1024)
                gen_toks = generate_tokens(am_model, caches, t.shape[1], logits, max_tokens=15, is_anchormass=True)
                out_text = sp.decode(gen_toks).strip()

                import re
                m = re.search(r"\d{4}", out_text)
                extracted = m.group(0) if m else ""
                hit = int(extracted == passkey)
                hits.append(hit)

                match_digits = sum(1 for a, b in zip(extracted[:4], passkey) if a == b)
                digits.append(match_digits)

            policy_hits[pol] = hits
            policy_digits[pol] = digits

        ada_hits = policy_hits["adaptive"]
        for pol in policies:
            h = policy_hits[pol]
            d = policy_digits[pol]
            corr = sum(h)
            lo, hi = wilson_score_interval(corr, total_trials)
            avg_d = float(np.mean(d))

            entry = {
                "correct": corr,
                "total": total_trials,
                "accuracy": corr / total_trials * 100.0,
                "ci95": [lo, hi],
                "avg_digits": avg_d,
            }

            if pol != "adaptive":
                b_only = sum(1 for a, p in zip(ada_hits, h) if a and not p)
                p_only = sum(1 for a, p in zip(ada_hits, h) if not a and p)
                p_val = paired_mcnemar_exact(b_only, p_only)
                entry["adaptive_only"] = b_only
                entry["other_only"] = p_only
                entry["mcnemar_p_vs_adaptive"] = p_val

            results[str(budget)][pol] = entry

    del am_model
    torch.cuda.empty_cache()
    return results


def main():
    parser = argparse.ArgumentParser(description="Run Supplementary Evaluations for AnchorMass-150M")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--sp_model", type=str, default="data/tokenizer.model")
    parser.add_argument("--output", type=str, default="logs/supp_eval_ablation_and_jcqa.json")
    parser.add_argument("--eval_jcqa", action="store_true", help="Run JCommonsenseQA across 4 stages")
    parser.add_argument("--eval_ablation", action="store_true", help="Run 5-way cache policy ablation (n=30)")
    parser.add_argument("--eval_ppl", action="store_true", help="Run 10-anchor long-context PPL with paired bootstrap and Holm p-values")
    parser.add_argument("--eval_rope_check", action="store_true", help="Run 4K RoPE oneshot vs chunked consistency & decode session invariance check")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    sp = spm.SentencePieceProcessor()
    sp.load(str(ROOT_DIR / args.sp_model))

    out_file = ROOT_DIR / args.output
    out_file.parent.mkdir(parents=True, exist_ok=True)

    data = {}
    if out_file.exists():
        with open(out_file, "r", encoding="utf-8") as f:
            data = json.load(f)

    if args.eval_jcqa:
        print("\n=== Running JCommonsenseQA Across 4 Training Stages ===")
        jcqa_data_file = ROOT_DIR / "data/jcommonsenseqa_val.jsonl"
        with open(jcqa_data_file, "r", encoding="utf-8") as f:
            jcqa_dataset = [json.loads(line) for line in f]

        stage_candidates = [
            ("stage1_1k_pretrain",
             ["checkpoints/standard/standard_150M.pt", "checkpoints/baseline/tf_1_150M.pt"],
             ["checkpoints/anchormass/anchormass_150M.pt", "checkpoints/anchormass/lb_11_150M.pt"]),
            ("stage2_4k_warmup",
             ["checkpoints/standard/standard_150M_4k_warmup.pt", "checkpoints/baseline/tf_1_150M_4k_warmup.pt"],
             ["checkpoints/anchormass/anchormass_150M_4k_warmup.pt", "checkpoints/anchormass/lb_11_150M_4k_warmup.pt"]),
            ("stage3a_32k_sft",
             ["checkpoints/standard/v2/standard_150M_32k_sft.pt", "checkpoints/baseline/v2/tf_1_150M_32k_sft.pt"],
             ["checkpoints/anchormass/v2/anchormass_150M_32k_sft.pt", "checkpoints/anchormass/v2/lb_11_150M_32k_sft.pt"]),
            ("stage3b_chat_sft",
             ["checkpoints/standard/standard_150M_chat_sft.pt", "checkpoints/baseline/tf_1_150M_chat_sft.pt"],
             ["checkpoints/anchormass/anchormass_150M_chat_sft.pt", "checkpoints/anchormass/lb_11_150M_chat_sft.pt"]),
        ]

        data["jcommonsenseqa"] = {}
        for sname, b_paths, l_paths in stage_candidates:
            b_pt = next((ROOT_DIR / p for p in b_paths if (ROOT_DIR / p).exists()), ROOT_DIR / b_paths[0])
            l_pt = next((ROOT_DIR / p for p in l_paths if (ROOT_DIR / p).exists()), ROOT_DIR / l_paths[0])
            print(f"\n--- {sname} ---")
            data["jcommonsenseqa"][sname] = {
                "n200": eval_stage_jcqa(sp, args.device, b_pt, l_pt, jcqa_dataset, limit=200),
                "n1119": eval_stage_jcqa(sp, args.device, b_pt, l_pt, jcqa_dataset, limit=None),
            }

    if args.eval_ablation:
        print("\n=== Running 5-Way Cache Policy Ablation (8K, 5 depths x 6 trials = 30 paired trials) ===")
        am_ckpt = ROOT_DIR / "checkpoints/anchormass/v2/anchormass_150M_32k_sft.pt"
        if not am_ckpt.exists():
            am_ckpt = ROOT_DIR / "checkpoints/anchormass/v2/lb_11_150M_32k_sft.pt"
        data["ablation_5way_n30"] = eval_ablation_5way_n30(sp, args.device, am_ckpt, seed=args.seed)

    if args.eval_ppl or args.eval_rope_check:
        from eval.long_context import (
            load_standard_model as load_std_sft,
            load_anchormass_model as load_am_sft,
            prefill_standard_oneshot,
        )
        from eval.utils import (
            prefill_standard_zero_oom,
            prefill_anchormass,
            generate_tokens,
        )
        from scipy.stats import wilcoxon, ttest_rel

        dev = torch.device(args.device)
        s_sft_path = ROOT_DIR / "checkpoints/standard/v2/standard_150M_32k_sft.pt"
        if not s_sft_path.exists():
            s_sft_path = ROOT_DIR / "checkpoints/baseline/v2/tf_1_150M_32k_sft.pt"
        a_sft_path = ROOT_DIR / "checkpoints/anchormass/v2/anchormass_150M_32k_sft.pt"
        if not a_sft_path.exists():
            a_sft_path = ROOT_DIR / "checkpoints/anchormass/v2/lb_11_150M_32k_sft.pt"
        s_sft, _ = load_std_sft(dev, torch.bfloat16, str(s_sft_path))
        a_sft, _ = load_am_sft(dev, torch.bfloat16, str(a_sft_path))
        val_tokens = np.memmap(str(ROOT_DIR / "data/corpus/token/val.bin"), dtype=np.uint16, mode="r")

        if args.eval_ppl:
            print("\n=== Running 10-Anchor Long-Context PPL with Paired Bootstrap & 6-Length Holm Adjustment ===")
            max_p = 32768
            usable = len(val_tokens) - max_p - 10
            step = usable // 10
            anchors = [max_p + i * step for i in range(10)]
            ppl_res = {}
            raw_pvals = []
            seq_lens = [1024, 2048, 4096, 8192, 16384, 32768]
            for slen in seq_lens:
                s_losses, a_losses = [], []
                for anc in anchors:
                    cx = torch.from_numpy(val_tokens[anc - slen : anc].astype(np.int64)).unsqueeze(0).to(dev)
                    cy = torch.from_numpy(val_tokens[anc - slen + 1 : anc + 1].astype(np.int64)).unsqueeze(0).to(dev)[:, -1024:]
                    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        _, slog = prefill_standard_oneshot(s_sft, cx, return_last_chunk_logits=True)
                        _, alog = prefill_anchormass(a_sft, cx, chunk_size=1024, return_last_chunk_logits=True)
                    sl = F.cross_entropy(slog.float().view(-1, slog.size(-1)), cy.reshape(-1)).item()
                    al = F.cross_entropy(alog.float().view(-1, alog.size(-1)), cy.reshape(-1)).item()
                    s_losses.append(sl)
                    a_losses.append(al)
                s_arr, a_arr = np.array(s_losses), np.array(a_losses)
                rng = np.random.default_rng(args.seed)
                idx = rng.integers(0, 10, size=(10000, 10))
                s_boot = np.exp(s_arr[idx].mean(axis=1))
                a_boot = np.exp(a_arr[idx].mean(axis=1))
                d_boot = a_boot - s_boot
                wp = float(wilcoxon(s_arr, a_arr).pvalue)
                tp = float(ttest_rel(s_arr, a_arr).pvalue)
                raw_pvals.append((str(slen), wp))
                ppl_res[str(slen)] = {
                    "std_corpus_ppl": float(np.exp(s_arr.mean())),
                    "std_loss_mean": float(s_arr.mean()),
                    "std_loss_sd": float(s_arr.std(ddof=1)),
                    "std_ppl_boot_ci95": [float(np.percentile(s_boot, 2.5)), float(np.percentile(s_boot, 97.5))],
                    "base_corpus_ppl": float(np.exp(s_arr.mean())),
                    "base_loss_mean": float(s_arr.mean()),
                    "base_loss_sd": float(s_arr.std(ddof=1)),
                    "base_ppl_boot_ci95": [float(np.percentile(s_boot, 2.5)), float(np.percentile(s_boot, 97.5))],
                    "am_corpus_ppl": float(np.exp(a_arr.mean())),
                    "am_loss_mean": float(a_arr.mean()),
                    "am_loss_sd": float(a_arr.std(ddof=1)),
                    "am_ppl_boot_ci95": [float(np.percentile(a_boot, 2.5)), float(np.percentile(a_boot, 97.5))],
                    "diff_ppl_boot_ci95": [float(np.percentile(d_boot, 2.5)), float(np.percentile(d_boot, 97.5))],
                    "am_wins_out_of_10": int((a_arr < s_arr).sum()),
                    "wilcoxon_p": wp,
                    "paired_t_p": tp,
                    "per_anchor_std_loss": [round(x, 4) for x in s_losses],
                    "per_anchor_base_loss": [round(x, 4) for x in s_losses],
                    "per_anchor_am_loss": [round(x, 4) for x in a_losses],
                }
            sorted_p = sorted(raw_pvals, key=lambda x: x[1])
            m = len(sorted_p)
            running_max = 0.0
            for rank_idx, (k_len, p_val) in enumerate(sorted_p):
                adj = min(1.0, p_val * (m - rank_idx))
                running_max = max(running_max, adj)
                ppl_res[k_len]["wilcoxon_holm_p6"] = round(running_max, 6)
            data["long_context_ppl_10anchors"] = ppl_res

        if args.eval_rope_check:
            print("\n=== Running 4K RoPE Oneshot vs Chunked Consistency & Decode Invariance Check ===")
            seq = torch.from_numpy(val_tokens[10000:14096].astype(np.int64)).unsqueeze(0).to(dev)
            target = torch.from_numpy(val_tokens[10001:14097].astype(np.int64)).unsqueeze(0).to(dev)[:, -1024:]
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                _, logits_one = prefill_standard_oneshot(s_sft, seq, return_last_chunk_logits=True)
                _, logits_chk = prefill_standard_zero_oom(s_sft, seq, chunk_size=1024, return_last_chunk_logits=True)
            diff = (logits_one.float() - logits_chk.float()).abs()
            top1_one, top1_chk = logits_one.argmax(dim=-1), logits_chk.argmax(dim=-1)
            cos_sim = F.cosine_similarity(logits_one.float(), logits_chk.float(), dim=-1).mean().item()
            loss_one = F.cross_entropy(logits_one.float().view(-1, logits_one.size(-1)), target.reshape(-1)).item()
            loss_chk = F.cross_entropy(logits_chk.float().view(-1, logits_chk.size(-1)), target.reshape(-1)).item()

            seq_8k = torch.from_numpy(val_tokens[10000:18192].astype(np.int64)).unsqueeze(0).to(dev)
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                caches, first_logits = prefill_anchormass(a_sft, seq_8k, chunk_size=1024)
                cos_b = a_sft.layers[0].attn.rope.cos_cached.clone()
                s_b = a_sft.layers[0].attn.rope.session_len
                _ = generate_tokens(a_sft, caches, seq_8k.shape[1], first_logits, max_tokens=15, is_anchormass=True, eos_ids=())
                cos_a = a_sft.layers[0].attn.rope.cos_cached.clone()
                s_a = a_sft.layers[0].attn.rope.session_len
                c_a = a_sft.layers[0].attn.rope.cached_len
            data["rope_oneshot_vs_chunked_4k"] = {
                "max_logit_diff": round(float(diff.max().item()), 6),
                "mean_logit_diff": round(float(diff.mean().item()), 6),
                "top1_token_match_rate": round(float((top1_one == top1_chk).float().mean().item()) * 100.0, 4),
                "top1_matches_out_of_1024": int((top1_one == top1_chk).sum().item()),
                "mean_cosine_similarity": round(cos_sim, 8),
                "oneshot_loss": round(loss_one, 6),
                "chunked_loss": round(loss_chk, 6),
                "abs_loss_diff": round(abs(loss_one - loss_chk), 6),
                "decode_session_base_invariant_8k_15toks": bool(torch.equal(cos_b, cos_a) and s_b == s_a == 8192),
                "session_len_locked": int(s_a),
                "cached_table_len": int(c_a),
            }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"\n[DONE] Supplementary evaluation results saved to: {out_file}")


if __name__ == "__main__":
    main()
