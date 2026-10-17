import os
import sys
import math
import json
import time
import random
import argparse
from pathlib import Path
import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import sentencepiece as spm

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from anchormass.model import AnchorMassForCausalLM, AnchorMassConfig
from eval.utils import prefill_anchormass, generate_tokens, get_haystack_tokens

torch.set_float32_matmul_precision("high")


class LongNeedleSFTDataset(Dataset):
    def __init__(self, jsonl_path, sp, max_seq_len=8192):
        self.sp = sp
        self.max_seq_len = max_seq_len
        self.samples = []

        print(f"Loading Long SFT data from: {jsonl_path}")
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.samples.append(json.loads(line))
        print(f"Loaded {len(self.samples)} samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        prompt_text = item["instruction"]
        answer_text = item["output"]

        prompt_ids = self.sp.encode(prompt_text, out_type=int)
        answer_ids = self.sp.encode(answer_text, out_type=int) + [self.sp.eos_id()]

        if len(prompt_ids) + len(answer_ids) > self.max_seq_len:
            prompt_ids = prompt_ids[-(self.max_seq_len - len(answer_ids)):]

        input_ids = prompt_ids + answer_ids
        labels = [-100] * len(prompt_ids) + answer_ids

        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


def collate_fn(batch):
    max_len = max(len(x[0]) for x in batch)
    input_ids_list = []
    labels_list = []
    for x, y in batch:
        pad_len = max_len - len(x)
        if pad_len > 0:
            x_padded = F.pad(x, (0, pad_len), value=0)
            y_padded = F.pad(y, (0, pad_len), value=-100)
        else:
            x_padded = x
            y_padded = y
        input_ids_list.append(x_padded)
        labels_list.append(y_padded)
    return torch.stack(input_ids_list), torch.stack(labels_list)


def set_model_cache_and_policy(model, cache_budget, policy="adaptive"):
    for layer in model.layers:
        a = getattr(layer, "_orig_mod", layer).attn
        a.max_global_cache = cache_budget
        a.eviction_policy = policy


@torch.no_grad()
def run_eval_needle(model, sp, ctx_len, device, depth=0.7):
    model.eval()
    passkey = str(random.randint(1000, 9999))
    needle_text = f"\nユーザー: 重要機密事項です。今回の秘密パスキーは【{passkey}】です。忘れないでください。\nアシスタント: 了解いたしました。秘密パスキー【{passkey}】を記憶しました。\n"
    query_text = "ユーザー: 先ほど記録した秘密パスキーは何ですか？余計な解説は一切含めず、4桁の半角数字のみで答えてください。\nアシスタント: "

    needle_ids = sp.encode(needle_text, out_type=int)
    query_ids = sp.encode(query_text, out_type=int)

    haystack_len = ctx_len - len(needle_ids) - len(query_ids)
    haystack = get_haystack_tokens(haystack_len, sp)

    insert_idx = int(len(haystack) * depth)
    full_ids = haystack[:insert_idx] + needle_ids + haystack[insert_idx:] + query_ids
    t = torch.tensor([full_ids], device=device)

    caches, logits = prefill_anchormass(model, t, chunk_size=1024)
    gen_toks = generate_tokens(model, caches, t.shape[1], logits, max_tokens=10, is_anchormass=True)
    out_text = sp.decode(gen_toks).strip()

    hit = (out_text == passkey)
    model.train()
    return passkey, out_text, hit


def train_stage(model, cfg, sp, device, data_path, save_path, max_seq_len, epochs, lr, max_cache, grad_accum=8):
    print(f"\nLong Sft: Max Len {max_seq_len:,} | Cache: {max_cache} | LR: {lr:.1e} | Epochs: {epochs}\n")

    set_model_cache_and_policy(model, cache_budget=max_cache, policy="adaptive")
    model.set_session_length(max_seq_len)

    from transformers.optimization import Adafactor
    optimizer = Adafactor(
        model.parameters(),
        lr=lr,
        scale_parameter=False,
        relative_step=False,
        warmup_init=False,
        weight_decay=0.01,
    )

    dataset = LongNeedleSFTDataset(ROOT_DIR / data_path, sp, max_seq_len=max_seq_len)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)

    total_steps = (len(loader) // grad_accum) * epochs

    print("\n--- Needle Check (Before) ---")
    for check_len in [2048, min(max_seq_len, 4096), max_seq_len]:
        p, o, h = run_eval_needle(model, sp, check_len, device)
        print(f"  [{check_len//1024}K] Target: 【{p}】 | Output: \"{o[:25]}\" -> {'Success' if h else 'Failed'}")

    model.set_session_length(max_seq_len)

    for layer in model.layers:
        if not getattr(layer, "_is_ckpt_wrapped", False):
            orig_f = layer.forward
            def make_ckpt(f):
                def wrapper(*layer_args, **layer_kwargs):
                    return torch.utils.checkpoint.checkpoint(f, *layer_args, use_reentrant=False, **layer_kwargs)
                return wrapper
            layer.forward = make_ckpt(orig_f)
            layer._is_ckpt_wrapped = True

    model.train()
    step = 0
    t0 = time.time()

    t_chunk = 2048
    C = getattr(cfg, "chunk_size", 64)
    sink = getattr(cfg, "sink_tokens", 16)
    first_len = sink + max(1, (t_chunk - sink) // C) * C
    step_len = max(1, t_chunk // C) * C

    for epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0

        for micro, (x, y) in enumerate(loader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            B, T = x.shape

            shift_labels = y[:, 1:].contiguous()

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                caches = [None] * len(model.layers)
                pos = 0
                h_slices = []
                target_slices = []

                while pos < T:
                    end = min(pos + (first_len if pos == 0 else step_len), T)
                    chunk_len = end - pos
                    h = model.embed(x[:, pos:end])
                    for i, layer in enumerate(model.layers):
                        h, caches[i] = layer(h, kv_cache=caches[i], offset=pos, use_cache=True)

                    valid_end = chunk_len if end < T else chunk_len - 1
                    chunk_targets = shift_labels[:, pos : pos + valid_end]
                    chunk_h = h[:, :valid_end, :]

                    mask = (chunk_targets != -100)
                    if mask.any():
                        h_norm = model.final_norm(chunk_h)
                        h_slices.append(h_norm[mask])
                        target_slices.append(chunk_targets[mask])
                    pos = end

                if len(h_slices) > 0:
                    selected_h = torch.cat(h_slices, dim=0)
                    selected_targets = torch.cat(target_slices, dim=0)
                    logits = model.output_proj(selected_h)
                    loss = F.cross_entropy(logits, selected_targets)
                else:
                    loss = torch.tensor(0.0, device=device, requires_grad=True)

                loss = loss / grad_accum

            loss.backward()
            accum_loss += loss.item() * grad_accum

            if (micro + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1

                peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 3)
                print(f"  [Len {max_seq_len//1024}K] Step {step:2d}/{total_steps} | Loss: {accum_loss:.4f} | VRAM: {peak_vram:.2f} GB", flush=True)
                accum_loss = 0.0

        save_p = ROOT_DIR / save_path
        save_p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), str(save_p))
        print(f"Epoch {epoch+1} Weight saved: {save_p}", flush=True)

    elapsed_m = (time.time() - t0) / 60
    print(f"\nStage finish Time: {elapsed_m:.1f} mins")
    print("--- Needle Check (After) ---")
    for check_len in [2048, min(max_seq_len, 4096), max_seq_len]:
        p, o, h = run_eval_needle(model, sp, check_len, device)
        print(f"  [{check_len//1024}K] Target: 【{p}】 | Output: \"{o[:25]}\" -> {'Success' if h else 'Failed'}")


def main():
    parser = argparse.ArgumentParser(description="AnchorMass Curriculum SFT (8K -> 16K -> 32K)")
    parser.add_argument("--model", type=str, default="anchormass_150M", help="Model key in config.yaml")
    parser.add_argument("--init_from", type=str, default=None, help="Initial weights path")
    parser.add_argument("--start_stage", type=str, default="8k", choices=["8k", "16k", "32k"])
    parser.add_argument("--data_8k", type=str, default="data/sft_150M_8k.jsonl")
    parser.add_argument("--data_16k", type=str, default="data/sft_150M_16k.jsonl")
    parser.add_argument("--data_32k", type=str, default="data/sft_150M_32k.jsonl")
    parser.add_argument("--epochs_8k", type=int, default=6)
    parser.add_argument("--epochs_16k", type=int, default=6)
    parser.add_argument("--epochs_32k", type=int, default=6)
    parser.add_argument("--output_dir", type=str, default="models")
    parser.add_argument("--grad_accum", type=int, default=8)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sp_path = ROOT_DIR / "tokenizer/tokenizer.model"
    sp = spm.SentencePieceProcessor(model_file=str(sp_path))

    with open(ROOT_DIR / "anchormass/config.yaml") as f:
        cfg_dict = yaml.safe_load(f)[args.model]

    cfg_dict["SEQ_LEN"] = 32768
    cfg_dict["MAX_POSITION_EMBEDDINGS"] = 262144
    cfg_dict["ROPE_SCALING_TYPE"] = "dynamic_ntk"
    cfg = AnchorMassConfig.from_dict(cfg_dict)

    model = AnchorMassForCausalLM(cfg).to(device=device, dtype=torch.bfloat16)
    print(f"  Gradient Checkpointing ready across {len(model.layers)} layers")

    out_dir = ROOT_DIR / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    m_name = args.model

    if args.init_from is not None:
        init_path = ROOT_DIR / args.init_from
    elif args.start_stage == "8k":
        default_init = cfg_dict.get("MODEL_SAVE_PATH", f"models/{m_name}.pt")
        init_path = ROOT_DIR / default_init
    elif args.start_stage == "16k":
        init_path = out_dir / f"{m_name}_8k_sft.pt"
    else:
        init_path = out_dir / f"{m_name}_16k_sft.pt"

    print(f"Loading initial weights from: {init_path}")
    sd = torch.load(init_path, map_location=device, weights_only=False)
    if isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    sd = {k.replace("._orig_mod.", ".").removeprefix("_orig_mod."): v for k, v in sd.items()}
    sd = {k: v for k, v in sd.items() if not any(x in k for x in ("cos_cached", "sin_cached", "inv_freq"))}
    model.load_state_dict(sd, strict=False)

    stages = [
        ("8k", args.data_8k, str(out_dir / f"{m_name}_8k_sft.pt"), 8192, args.epochs_8k, 3.0e-5, 2048),
        ("16k", args.data_16k, str(out_dir / f"{m_name}_16k_sft.pt"), 16384, args.epochs_16k, 2.5e-5, 4096),
        ("32k", args.data_32k, str(out_dir / f"{m_name}_32k_sft.pt"), 32768, args.epochs_32k, 2.0e-5, 8192),
    ]

    start_idx = 0
    for i, s in enumerate(stages):
        if s[0] == args.start_stage:
            start_idx = i
            break

    total_start = time.time()
    for name, data_p, save_p, max_len, epochs, lr, max_cache in stages[start_idx:]:
        train_stage(model, cfg, sp, device, data_p, save_p, max_len, epochs, lr, max_cache, grad_accum=args.grad_accum)

    total_elapsed = (time.time() - total_start) / 60
    final_model = out_dir / f"{m_name}_32k_sft.pt"
    print(f"Long Sft finish Time: {total_elapsed:.1f} mins")
    print(f"Model saved: {final_model}")


if __name__ == "__main__":
    main()
