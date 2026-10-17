import os
import sys
import math
import time
import json
import random
import yaml
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import sentencepiece as spm
from datasets import load_dataset

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from anchormass.standard_model import GPTLanguageModel, GPTConfig
from anchormass.model import AnchorMassForCausalLM, AnchorMassConfig
from eval.utils import expand_standard_rope

torch.set_float32_matmul_precision("high")


SYSTEM_PROMPT = "あなたは親切なAIアシスタントです。指示に対して自然で正確な日本語で回答してください。"


def format_chat_prompt(instruction: str, input_info: str = "") -> str:
    if input_info:
        return f"System: {SYSTEM_PROMPT}\n\nユーザー: {instruction}\n参考情報: {input_info}\n\nアシスタント: "
    return f"System: {SYSTEM_PROMPT}\n\nユーザー: {instruction}\n\nアシスタント: "


class JapaneseChatDataset(Dataset):
    def __init__(self, sp, max_samples=8000, max_seq_len=512, seed=42):
        self.sp = sp
        self.max_seq_len = max_seq_len
        self.samples = []

        print(f"Loading Japanese Alpaca dataset from cache/hub...")
        ds = load_dataset("fujiki/japanese_alpaca_data", split="train")

        raw_items = list(ds)
        rng = random.Random(seed)
        rng.shuffle(raw_items)

        for item in raw_items:
            inst = (item.get("instruction") or "").strip()
            out = (item.get("output") or "").strip()
            inp = (item.get("input") or "").strip()

            if not inst or not out:
                continue
            if len(inst) < 5 or len(inst) > 200 or len(out) < 8 or len(out) > 400:
                continue
            if inp and len(inp) > 150:
                continue
            if inst in out and len(out) <= len(inst) + 5:
                continue

            prompt_text = format_chat_prompt(inst, inp)
            full_text = prompt_text + out

            full_ids = self.sp.encode(full_text, out_type=int) + [self.sp.eos_id()]
            prompt_ids = self.sp.encode(prompt_text, out_type=int)

            p_len = len(prompt_ids)
            if p_len >= len(full_ids) or len(full_ids) > self.max_seq_len:
                continue

            labels = [-100] * p_len + full_ids[p_len:]

            self.samples.append({
                "input_ids": full_ids,
                "labels": labels,
            })
            if len(self.samples) >= max_samples:
                break

        print(f"Collected {len(self.samples)} clean chat samples (max_len={max_seq_len}).")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        input_ids = list(item["input_ids"])
        labels = list(item["labels"])

        pad_len = self.max_seq_len - len(input_ids)
        input_ids = input_ids + [self.sp.pad_id()] * pad_len
        labels = labels + [-100] * pad_len

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def load_standard_model(device, ckpt_path):
    print(f"\n[Loading Standard 150M from {ckpt_path}]")
    with open(ROOT_DIR / "anchormass/config.yaml") as f:
        b_cfg = yaml.safe_load(f)["standard_150M"]
    gpt_cfg = GPTConfig(
        vocab_size=b_cfg["VOCAB_SIZE"], dim=b_cfg["DIM"], ff_dim=b_cfg.get("FF_DIM", b_cfg["DIM"] * 4),
        num_heads=b_cfg["NUM_HEADS"], num_kv_heads=b_cfg["NUM_KV_HEADS"], num_layers=b_cfg["NUM_LAYERS"],
        seq_len=b_cfg.get("SEQ_LEN", 1024), dropout=0.0
    )
    model = GPTLanguageModel(gpt_cfg).to(device=device, dtype=torch.bfloat16)
    sd = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    sd = {k.replace("._orig_mod.", ".").removeprefix("_orig_mod."): v for k, v in sd.items()}
    sd = {k: v for k, v in sd.items() if not any(x in k for x in ("cos_cached", "sin_cached", "inv_freq"))}
    model.load_state_dict(sd, strict=False)
    expand_standard_rope(model, max_len=16384)
    return model, b_cfg["VOCAB_SIZE"]


def load_anchormass_model(device, ckpt_path):
    print(f"\n[Loading AnchorMass-150M from {ckpt_path}]")
    with open(ROOT_DIR / "anchormass/config.yaml") as f:
        l_cfg = yaml.safe_load(f)["anchormass_150M"]
    l_cfg["MAX_GLOBAL_CACHE"] = 8192
    l_cfg["MAX_POSITION_EMBEDDINGS"] = 262144
    l_cfg["ROPE_SCALING_TYPE"] = "dynamic_ntk"
    l_cfg["EVICTION_POLICY"] = "adaptive"
    cfg = AnchorMassConfig.from_dict(l_cfg)
    model = AnchorMassForCausalLM(cfg).to(device=device, dtype=torch.bfloat16)
    sd = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    sd = {k.replace("._orig_mod.", ".").removeprefix("_orig_mod."): v for k, v in sd.items()}
    sd = {k: v for k, v in sd.items() if not any(x in k for x in ("cos_cached", "sin_cached", "inv_freq"))}
    model.load_state_dict(sd, strict=False)
    return model, l_cfg["VOCAB_SIZE"]


def get_cosine_lr(step, total_steps, warmup_steps, max_lr, min_lr):
    if step < warmup_steps:
        return max_lr * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def main():
    parser = argparse.ArgumentParser(description="150M Controlled Chat SFT Training")
    parser.add_argument("--model", type=str, required=True, choices=["standard", "anchormass"])
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min_lr", type=float, default=1e-5)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--max_samples", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sp_path = ROOT_DIR / "tokenizer/tokenizer.model"
    sp = spm.SentencePieceProcessor(model_file=str(sp_path))

    dataset = JapaneseChatDataset(sp, max_samples=args.max_samples, max_seq_len=512, seed=args.seed)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    if args.model == "standard":
        ckpt_in = ROOT_DIR / "checkpoints/standard/standard_150M_4k_warmup.pt"
        if not ckpt_in.exists():
            ckpt_in = ROOT_DIR / "checkpoints/standard/tf_1_150M_4k_warmup.pt"
        if not ckpt_in.exists():
            ckpt_in = ROOT_DIR / "checkpoints/baseline/tf_1_150M_4k_warmup.pt"
        ckpt_out = ROOT_DIR / "checkpoints/standard/standard_150M_chat_sft.pt"
        model, vocab_size = load_standard_model(device, str(ckpt_in))
    else:
        ckpt_in = ROOT_DIR / "checkpoints/anchormass/anchormass_150M_4k_warmup.pt"
        if not ckpt_in.exists():
            ckpt_in = ROOT_DIR / "checkpoints/anchormass/lb_11_150M_4k_warmup.pt"
        ckpt_out = ROOT_DIR / "checkpoints/anchormass/anchormass_150M_chat_sft.pt"
        model, vocab_size = load_anchormass_model(device, str(ckpt_in))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95))

    print(f"Start chat sft: {args.model.upper()} 150M")
    print(f"Base Checkpoint : {ckpt_in}")
    print(f"Target Output   : {ckpt_out}")
    print(f"Steps: {args.steps} | Batch: {args.batch_size} x {args.grad_accum} = {args.batch_size * args.grad_accum} | LR: {args.lr}")

    model.train()
    optimizer.zero_grad(set_to_none=True)

    step = 0
    t0 = time.time()
    accum_loss = 0.0
    accum_toks = 0
    epoch = 0

    while step < args.steps:
        epoch += 1
        for batch in dataloader:
            if step >= args.steps:
                break

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            x = input_ids[:, :-1].contiguous()
            y = labels[:, 1:].contiguous()

            lr = get_cosine_lr(step, args.steps, args.warmup_steps, args.lr, args.min_lr)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(x)
                loss = F.cross_entropy(
                    logits.reshape(-1, vocab_size),
                    y.reshape(-1),
                    ignore_index=-100,
                    reduction="sum"
                )
                valid_tokens = (y != -100).sum()
                loss_scaled = loss / (valid_tokens.clamp_min(1) * args.grad_accum)

            loss_scaled.backward()
            accum_loss += loss.item()
            accum_toks += valid_tokens.item()

            if (step + 1) % args.grad_accum == 0 or (step + 1) == args.steps:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            step += 1

            if step % 20 == 0 or step == args.steps:
                avg_loss = accum_loss / max(1, accum_toks)
                sec_per_step = (time.time() - t0) / step
                eta_sec = (args.steps - step) * sec_per_step
                vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
                print(f"  Step {step:4d}/{args.steps} | Loss: {avg_loss:6.3f} | LR: {lr:.2e} | VRAM: {vram_gb:4.1f} GB | Speed: {sec_per_step:.3f} s/step | ETA: {eta_sec/60:4.1f} min")

    print(f"\n[Saving Chat SFT weights to: {ckpt_out}]")
    ckpt_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "step": step,
        "final_loss": avg_loss,
        "config": args.model,
    }, ckpt_out)
    print("Successfully saved model checkpoint.")


if __name__ == "__main__":
    main()
