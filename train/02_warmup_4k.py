import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import yaml

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.set_float32_matmul_precision("high")

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from anchormass.model import AnchorMassForCausalLM, AnchorMassConfig


class PrefetchDataStream:
    def __init__(self, bin_path, seq_len=4096, seed=42, shuffle=True):
        self.data = np.memmap(bin_path, dtype=np.uint16, mode="r")
        self.total_tokens = len(self.data)
        self.seq_len = seq_len
        self.chunk_len = seq_len + 1
        self.shuffle = shuffle
        self.rng = np.random.RandomState(seed)
        self.max_idx = max(1, self.total_tokens - self.chunk_len)
        self.current_idx = 0

    def next_sample(self, device):
        if self.shuffle:
            start = self.rng.randint(0, self.max_idx)
        else:
            start = self.current_idx
            self.current_idx = (self.current_idx + self.chunk_len) % self.max_idx

        raw = self.data[start : start + self.chunk_len].astype(np.int64)
        tensor = torch.from_numpy(raw).to(device=device, non_blocking=True)
        input_ids = tensor[:-1].unsqueeze(0)
        targets = tensor[1:].unsqueeze(0)
        return input_ids, targets


def get_lr(step, warmup_steps, total_steps, peak_lr, min_lr):
    if step < warmup_steps:
        return min_lr + (peak_lr - min_lr) * (step / max(1, warmup_steps))
    progress = (step - warmup_steps) / max(1, (total_steps - warmup_steps))
    progress = min(max(progress, 0.0), 1.0)
    return min_lr + 0.5 * (peak_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate_val_loss(model, val_stream, device, num_samples=25, train_chunk_size=2048):
    model.eval()
    losses = []
    for _ in range(num_samples):
        x, y = val_stream.next_sample(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(x, train_chunk_size=train_chunk_size)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        losses.append(loss.item())
    model.train()
    mean_loss = float(np.mean(losses))
    ppl = math.exp(min(mean_loss, 20.0))
    return mean_loss, ppl


def main():
    parser = argparse.ArgumentParser(description="AnchorMass-150M 4k Warmup Pretraining")
    parser.add_argument("--steps", type=int, default=2000, help="Total training steps")
    parser.add_argument("--grad_accum_steps", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--peak_lr", type=float, default=6.0e-5, help="Peak learning rate")
    parser.add_argument("--min_lr", type=float, default=6.0e-6, help="Minimum learning rate")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Warmup steps")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--train_bin", type=str, default="data/corpus/token/part_04.bin", help="Training corpus bin")
    parser.add_argument("--val_bin", type=str, default="data/corpus/token/val.bin", help="Validation corpus bin")
    parser.add_argument("--init_from", type=str, default=None, help="Initial checkpoint path")
    parser.add_argument("--save_path", type=str, default="checkpoints/anchormass/anchormass_150M_4k_warmup.pt", help="Save checkpoint path")
    parser.add_argument("--log_file", type=str, default="runs/warmup_4k/anchormass_150M.jsonl", help="Progress log file")
    parser.add_argument("--val_every", type=int, default=200, help="Validation interval")
    parser.add_argument("--train_chunk_size", type=int, default=2048, help="Chunk size for AnchorMass forward")
    parser.add_argument("--hybrid_train_prob", type=float, default=0.25, help="Probability of priority mode during training")
    parser.add_argument("--eviction_policy", type=str, default="adaptive", help="Eviction policy (adaptive, merge, priority)")
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    print("\nAnchorMass-150M: 4096 系列長 追加事前学習 (Warmup)")
    print(f"Device: {torch.cuda.get_device_name(0)} | Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print(f"Steps: {args.steps} | Grad Accum: {args.grad_accum_steps} | Peak LR: {args.peak_lr:.2e} | Seed: {args.seed}")
    print(f"AnchorMass Config: chunk={args.train_chunk_size}, policy={args.eviction_policy}, hybrid_prob={args.hybrid_train_prob}")
    print(f"Corpus: {args.train_bin} | Save Path: {args.save_path}\n")

    with open(ROOT_DIR / "anchormass/config.yaml") as f:
        l_cfg_dict = yaml.safe_load(f)["anchormass_150M"]

    l_cfg_dict["SEQ_LEN"] = 4096
    l_cfg_dict["MAX_GLOBAL_CACHE"] = 8192
    l_cfg_dict["MAX_POSITION_EMBEDDINGS"] = 262144
    l_cfg_dict["ROPE_SCALING_TYPE"] = "dynamic_ntk"
    l_cfg_dict["ORIGINAL_SEQ_LEN"] = 1024
    l_cfg_dict["EVICTION_POLICY"] = args.eviction_policy
    l_cfg_dict["HYBRID_TRAIN_PROB"] = args.hybrid_train_prob
    l_cfg_dict["TRAIN_CHUNK_SIZE"] = args.train_chunk_size

    l_cfg = AnchorMassConfig.from_dict(l_cfg_dict)
    model = AnchorMassForCausalLM(l_cfg).to(device=device, dtype=dtype)

    if args.init_from is not None:
        init_ckpt_path = ROOT_DIR / args.init_from
    else:
        init_ckpt_path = ROOT_DIR / "checkpoints/anchormass/anchormass_150M.pt"
        if not init_ckpt_path.exists():
            init_ckpt_path = ROOT_DIR / "checkpoints/anchormass/lb_11_150M.pt"

    if init_ckpt_path.exists():
        print(f"  Loading initial weights from {init_ckpt_path}...")
        sd = torch.load(init_ckpt_path, map_location=device, weights_only=False)
        sd = {k.replace("._orig_mod.", ".").removeprefix("_orig_mod."): v for k, v in sd.items()}
        sd = {k: v for k, v in sd.items() if not any(x in k for x in ("cos_cached", "sin_cached", "inv_freq"))}
        model.load_state_dict(sd, strict=False)
        print("  Initial weights successfully loaded.")
    else:
        raise FileNotFoundError(f"Initial checkpoint not found: {init_ckpt_path}")

    for layer in model.layers:
        orig = layer.forward
        def make_ckpt(f):
            def wrapper(*layer_args, **layer_kwargs):
                return torch.utils.checkpoint.checkpoint(f, *layer_args, use_reentrant=False, **layer_kwargs)
            return wrapper
        layer.forward = make_ckpt(orig)

    decay_params = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    nodecay_params = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    optim_groups = [
        {"params": decay_params, "weight_decay": args.weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=args.min_lr, betas=(0.9, 0.95), eps=1e-8)

    train_stream = PrefetchDataStream(ROOT_DIR / args.train_bin, seq_len=4096, seed=args.seed, shuffle=True)
    val_stream = PrefetchDataStream(ROOT_DIR / args.val_bin, seq_len=4096, seed=args.seed + 100, shuffle=False)

    log_path = ROOT_DIR / args.log_file
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists():
        log_path.unlink()

    print("  Calculating initial validation loss...")
    init_val_loss, init_val_ppl = evaluate_val_loss(model, val_stream, device, num_samples=25, train_chunk_size=args.train_chunk_size)
    print(f"  [Step 0] Initial Val Loss: {init_val_loss:.4f} | Val PPL: {init_val_ppl:.2f}")

    model.train()
    optimizer.zero_grad(set_to_none=True)

    accum_loss = 0.0
    start_time = time.time()
    step_t0 = time.time()

    for step in range(1, args.steps + 1):
        cur_lr = get_lr(step, args.warmup_steps, args.steps, args.peak_lr, args.min_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = cur_lr

        for micro_step in range(args.grad_accum_steps):
            x, y = train_stream.next_sample(device)
            with torch.autocast(device_type="cuda", dtype=dtype):
                logits = model(x, train_chunk_size=args.train_chunk_size)
                loss = F.cross_entropy(logits.reshape(-1, l_cfg.vocab_size), y.reshape(-1))
                loss = loss / args.grad_accum_steps

            loss.backward()
            accum_loss += loss.item()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if step % 20 == 0 or step == 1 or step == args.steps:
            torch.cuda.synchronize()
            elapsed_step = (time.time() - step_t0) / (20 if step > 1 else 1)
            step_t0 = time.time()
            vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
            print(
                f"Step {step:4d}/{args.steps} | LR: {cur_lr:.2e} | Train Loss: {accum_loss:.4f} | "
                f"Grad Norm: {grad_norm:.3f} | {elapsed_step * 1000:.0f} ms/step | VRAM: {vram_gb:.2f} GB"
            )

        if step % args.val_every == 0 or step == args.steps:
            val_loss, val_ppl = evaluate_val_loss(model, val_stream, device, num_samples=25, train_chunk_size=args.train_chunk_size)
            vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
            print(f"  >>> [Validation Step {step}] Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.2f}")

            log_entry = {
                "step": step,
                "lr": cur_lr,
                "train_loss": accum_loss,
                "val_loss": val_loss,
                "val_ppl": val_ppl,
                "grad_norm": float(grad_norm),
                "vram_gb": vram_gb,
                "elapsed_sec": time.time() - start_time,
            }
            with open(log_path, "a") as f:
                f.write(json.dumps(log_entry) + "\n")

        accum_loss = 0.0

    save_file = ROOT_DIR / args.save_path
    save_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_file)
    print(f"\nAnchorMass-150M 4k Warmup completed in {time.time() - start_time:.1f}s")
    print(f"Saved checkpoint: {save_file}\n")


if __name__ == "__main__":
    main()
