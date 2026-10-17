import argparse
import math
import os
import random
import sys
import threading
import time
from typing import Optional

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import torch
import torch.nn.functional as F
import yaml
SummaryWriter = None

torch.set_float32_matmul_precision('high')

import sys
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from anchormass.model import AnchorMassForCausalLM, AnchorMassConfig
def clean_state_dict(sd):
    return {k.replace("._orig_mod.", ".").removeprefix("_orig_mod."): v for k, v in sd.items() if not any(x in k for x in ("cos_cached", "sin_cached", "inv_freq"))}
def ensure_experiment(*args, **kwargs): pass
def append_validation(*args, **kwargs): pass
def save_lab_checkpoint(*args, **kwargs): pass
def load_lab_config(*args, **kwargs): return {}

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "../anchormass/config.yaml")
VAL_STEPS = 20

class PrefetchSeqGenerator:
    def __init__(self, bin_path, batch_size, seq_len, device, start_step=0, grad_accum_steps=1, shuffle=True):
        self.data = np.memmap(bin_path, dtype=np.uint16, mode="r")
        self.total_tokens = len(self.data)
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.chunk_len = seq_len + 1
        self.device = device
        self.use_cuda = device.type == "cuda"
        self.shuffle = shuffle
        self.max_idx = max(1, self.total_tokens - self.chunk_len)

        segment_size = max(1, self.total_tokens // batch_size)
        consumed = (start_step * grad_accum_steps * self.chunk_len) % segment_size
        self.pointers = np.array(
            [i * segment_size + (consumed % max(1, (segment_size - self.chunk_len))) for i in range(batch_size)]
        )

        self._prefetch_batch = None
        self._prefetch_ready = threading.Event()
        self._prefetch_request = threading.Event()
        self._stop = False
        self._thread = threading.Thread(target=self._prefetch_loop, daemon=True)
        self._thread.start()
        self._prefetch_request.set()

        mode_str = "random-shuffle" if shuffle else "sequential"
        print(f"Streaming from {bin_path} ({self.total_tokens:,} tokens) [{mode_str}]")

    def _generate_one(self):
        if self.shuffle:
            starts = np.random.randint(0, self.max_idx, size=self.batch_size)
        else:
            for i in range(self.batch_size):
                if self.pointers[i] + self.chunk_len > self.total_tokens:
                    self.pointers[i] = 0
            starts = self.pointers.copy()
            self.pointers += self.chunk_len

        offsets = np.arange(self.chunk_len)
        idx_matrix = starts[:, None] + offsets[None, :]
        batch = torch.from_numpy(self.data[idx_matrix].astype(np.int32)).long()

        input_ids = batch[:, :-1]
        labels = batch[:, 1:]
        labels = labels.masked_fill(labels == 0, -100)

        if self.use_cuda:
            return input_ids.pin_memory(), labels.pin_memory()
        return input_ids, labels

    def _prefetch_loop(self):
        while not self._stop:
            self._prefetch_request.wait()
            self._prefetch_request.clear()
            if self._stop:
                break
            self._prefetch_batch = self._generate_one()
            self._prefetch_ready.set()

    def __next__(self):
        self._prefetch_ready.wait()
        self._prefetch_ready.clear()
        input_ids, labels = self._prefetch_batch
        self._prefetch_request.set()
        if self.use_cuda:
            input_ids = input_ids.to(self.device, non_blocking=True)
            labels    = labels.to(self.device, non_blocking=True)
        return input_ids, labels

    def __iter__(self):
        return self

    def stop(self):
        self._stop = True
        self._prefetch_request.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)


def get_lr(step, warmup_steps, total_steps, peak_lr, scheduler_type="wsd", stable_ratio=0.70):
    if step < warmup_steps:
        return peak_lr * float(step + 1) / max(1, warmup_steps)
    if scheduler_type == "wsd":
        stable_step = int(total_steps * stable_ratio)
        if step < stable_step:
            return peak_lr
        decay_steps = max(1, total_steps - stable_step)
        progress = min(1.0, float(step - stable_step) / decay_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return peak_lr * 0.05 + cosine * (peak_lr - peak_lr * 0.05)
    else:
        progress = min(1.0, float(step - warmup_steps) / max(1, total_steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return peak_lr * 0.1 + cosine * (peak_lr - peak_lr * 0.1)


def evaluate(model, val_gen, vocab_size, device):
    model.eval()
    total_loss = 0.0
    free_correct = 0
    free_total = 0
    offsets = [0, 384, 768, 1152, 1536]

    with torch.no_grad():
        for _ in range(VAL_STEPS):
            input_ids, labels = next(val_gen)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                logits = model(input_ids)
                lm_loss = F.cross_entropy(logits.reshape(-1, vocab_size), labels.reshape(-1), ignore_index=-100)
            total_loss += lm_loss.item()
            seq_len = input_ids.shape[1]
            valid_offs = [off for off in offsets if off + 128 <= seq_len]
            if not valid_offs:
                valid_offs = [0]

            prompts = torch.stack([input_ids[0, off : off + 64] for off in valid_offs], dim=0)
            targets = torch.stack([labels[0, off + 63 : off + 127] for off in valid_offs], dim=0)

            out_ids = model.generate(prompts, max_new_tokens=64, temperature=0.0, ignore_eos=True)
            gen_part = out_ids[:, 64:]
            gen_len = gen_part.shape[1]
            tgt_part = targets[:, :gen_len]

            valid = (tgt_part != -100)
            match = (gen_part == tgt_part) & valid
            free_correct += match.sum().item()
            free_total += valid.sum().item()

    model.train()
    return total_loss / VAL_STEPS, free_correct / max(1, free_total)


def save_checkpoint(model, optimizer, save_path, opt_path, experiment_id, step, val_loss, lab_cfg):
    cleaned = clean_state_dict(model)
    torch.save(cleaned, save_path)
    torch.save(optimizer.state_dict(), opt_path)
    save_lab_checkpoint(
        state_dict=cleaned,
        experiment_id=experiment_id,
        step=step,
        source_save_path=save_path,
        val_loss=val_loss,
        lab_config=lab_cfg,
    )
    print(f"Saved {save_path} ({len(cleaned)} keys)")


def main():
    parser = argparse.ArgumentParser(description="AnchorMass Pretrainer")
    parser.add_argument("--config", type=str, default=CONFIG_PATH)
    parser.add_argument("--model", type=str, default="anchormass_150M")
    parser.add_argument("--experiment_id", type=str, default=None, help="Experiment ID for lab runs")
    parser.add_argument("--steps", type=int, default=16000)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--lr_scale", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--grad_accum", type=int, default=None)
    parser.add_argument("--lr_scheduler", type=str, default="wsd", choices=["cosine", "wsd"])
    parser.add_argument("--stable_ratio", type=float, default=0.70)
    parser.add_argument("--train_bin", type=str, default="data/corpus/token/part_01.bin")
    parser.add_argument("--train_probs", type=str, default=None)
    parser.add_argument("--val_bin", type=str, default="data/corpus/token/val_5M.bin")
    parser.add_argument("--val_every", type=int, default=1000)
    parser.add_argument("--save_every", type=int, default=4000)
    parser.add_argument("--sample_every", type=int, default=4000)
    parser.add_argument("--start_step", type=int, default=0)
    parser.add_argument("--stop_step", type=int, default=None)
    parser.add_argument("--init_from", type=str, default=None, help="Path to checkpoint to initialize weights from")
    parser.add_argument("--no_compile", action="store_true")
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adafactor", "adamw"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    args = parser.parse_args()

    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    with open(args.config, "r", encoding="utf-8") as f:
        full_cfg = yaml.safe_load(f)
    params = full_cfg[args.model]

    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_type)

    if device_type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    sp.load("tokenizer/tokenizer.model")

    vocab_size = params.get("VOCAB_SIZE", 28000)
    seq_len = params.get("SEQ_LEN", 512)
    batch_size = args.batch_size if args.batch_size is not None else params.get("BATCH_SIZE", 8)
    grad_accum = args.grad_accum if args.grad_accum is not None else params.get("GRAD_ACCUM_STEPS", 16)
    peak_lr = float(params.get("PEAK_LEARNING_RATE", 6e-4)) * args.lr_scale
    weight_decay = params.get("WEIGHT_DECAY", 0.01)

    cfg = AnchorMassConfig.from_dict(params)
    model = AnchorMassForCausalLM(cfg).to(device)

    if args.init_from and os.path.exists(args.init_from):
        print(f"Initializing weights from: {args.init_from}")
        w = torch.load(args.init_from, map_location=device)
        model.load_state_dict(w, strict=False)
        print("Weights initialized successfully!")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"--- AnchorMass: {args.model} | {total_params/1e6:.1f}M params ---")
    print(f"Device: {device} | seq_len={seq_len} | chunk={cfg.chunk_size} | mem={cfg.mem_tokens} | batch={batch_size}x{grad_accum} ({batch_size*grad_accum*seq_len:,} tok/step)")

    train_model = model
    if not args.no_compile and device_type == "cuda":
        print("torch.compile: enabled")
        for i in range(len(model.layers)):
            model.layers[i] = torch.compile(model.layers[i])

    if args.optimizer == "adafactor":
        from transformers.optimization import Adafactor
        print("Optimizer: Adafactor")
        optimizer = Adafactor(
            model.parameters(),
            lr=peak_lr,
            scale_parameter=False,
            relative_step=False,
            warmup_init=False,
            weight_decay=weight_decay,
        )
    else:
        print("Optimizer: AdamW")
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=peak_lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.95),
            eps=1e-8,
        )


    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    model_save_path = os.path.join(root, params.get("MODEL_SAVE_PATH", f"models/{args.model}.pt"))
    optimizer_path  = model_save_path.replace(".pt", "_optimizer.pt")
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    experiment_id = args.experiment_id or f"am_{args.model}"
    lab_cfg = load_lab_config(root)
    ensure_experiment(
        experiment_id=experiment_id,
        args=args,
        arch_version="am",
        model_name=args.model,
        config_path=args.config,
        generate_entrypoint="demo.py",
        lab_config=lab_cfg,
    )

    train_bins  = [os.path.join(root, b.strip()) for b in args.train_bin.split(",")]
    if args.train_probs:
        raw_probs = [float(p) for p in args.train_probs.split(",")]
        if len(raw_probs) == len(train_bins):
            total_prob = sum(raw_probs)
            train_probs = [p / total_prob for p in raw_probs]
        else:
            print(f"[INFO] len(train_probs) ({len(raw_probs)}) != len(train_bins) ({len(train_bins)}). Using uniform sampling.")
            train_probs = [1.0 / len(train_bins)] * len(train_bins)
    else:
        train_probs = [1.0 / len(train_bins)] * len(train_bins)

    train_gens = [
        PrefetchSeqGenerator(b, batch_size=batch_size, seq_len=seq_len, device=device, shuffle=True)
        for b in train_bins
    ]

    val_gen = PrefetchSeqGenerator(
        os.path.join(root, args.val_bin),
        batch_size=batch_size,
        seq_len=seq_len,
        device=device,
        shuffle=False,
    )

    tb_dir = os.path.join(root, "runs", experiment_id)
    writer = SummaryWriter(log_dir=tb_dir) if SummaryWriter is not None else None

    use_wandb = False
    if args.wandb_project:
        try:
            import wandb
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=args.wandb_run_name or f"am_{args.model}",
                config=vars(args),
            )
            use_wandb = True
            print(f"[INFO] WandB initialized: project={args.wandb_project}, run={wandb.run.name}")
        except Exception as e:
            print(f"[WARN] Failed to initialize WandB: {e}. Continuing without WandB.")

    print(f"\nTraining: steps={args.steps}, peak_lr={peak_lr:.2e}, warmup={args.warmup_steps}")
    print(f"Val every {args.val_every} | Save every {args.save_every} | Sample every {args.sample_every}\n")

    last_val_loss = None

    try:
        for step in range(args.start_step, args.steps):
            if args.stop_step and step >= args.stop_step:
                break

            start = time.time()
            lr = get_lr(step, args.warmup_steps, args.steps, peak_lr, scheduler_type=args.lr_scheduler, stable_ratio=args.stable_ratio)

            for pg in optimizer.param_groups:
                pg["lr"] = lr

            optimizer.zero_grad(set_to_none=True)
            total_loss = 0.0

            for micro_step in range(grad_accum):
                gen = random.choices(train_gens, weights=train_probs, k=1)[0]
                input_ids, labels = next(gen)

                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits = train_model(input_ids)
                    lm_loss = F.cross_entropy(logits.reshape(-1, vocab_size), labels.reshape(-1), ignore_index=-100)
                    loss = lm_loss / grad_accum

                loss.backward()
                total_loss += loss.item()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            step_time = time.time() - start

            if writer:
                writer.add_scalar("Train/Loss", total_loss, step + 1)
                writer.add_scalar("Train/LR", lr, step + 1)
            if use_wandb:
                wandb.log({
                    "train/loss": total_loss,
                    "train/lr": lr,
                    "train/step_time": step_time,
                }, step=step + 1)

            if (step + 1) % 10 == 0 or step < 10:
                print(f"Step {step+1}/{args.steps} | loss={total_loss:.4f} | lr={lr:.2e} | {step_time:.2f}s")

            # Validation
            if (step + 1) % args.val_every == 0 or (step + 1) == args.steps:
                val_loss, free_accuracy = evaluate(model, val_gen, vocab_size, device)
                last_val_loss = val_loss
                if writer:
                    writer.add_scalar("Val/Loss", val_loss, step + 1)
                    writer.add_scalar("Val/Free_Accuracy", free_accuracy, step + 1)

                print(f"\nVAL Step {step+1} | val_loss(5M)={val_loss:.4f} | free_acc={free_accuracy:.4f}\n")

                if use_wandb:
                    log_data = {"val/loss": val_loss, "val/free_accuracy": free_accuracy}
                    if val_loss_2 is not None:
                        log_data["val/loss_2_indomain"] = val_loss_2
                    wandb.log(log_data, step=step + 1)

                metrics_dict = {"loss": val_loss, "free_accuracy": free_accuracy}
                if val_loss_2 is not None:
                    metrics_dict["val_loss_2"] = val_loss_2

                append_validation(
                    experiment_id,
                    step=step + 1,
                    metrics=metrics_dict,
                    lab_config=lab_cfg,
                )


            # Save
            if (step + 1) % args.save_every == 0 or (step + 1) == args.steps:
                save_checkpoint(model, optimizer, model_save_path, optimizer_path, experiment_id, step + 1, last_val_loss, lab_cfg)

            # Sample
            if args.sample_every and ((step + 1) % args.sample_every == 0 or (step + 1) == args.steps):
                model.eval()
                sample_dir = os.path.join(root, "experiments", "runs", experiment_id, "samples")
                os.makedirs(sample_dir, exist_ok=True)
                sample_file = os.path.join(sample_dir, f"step-{step+1:05d}.txt")

                test_prompt = "人工知能の発展により"
                prompt_ids = sp.encode_as_ids(test_prompt)
                p_tensor = torch.tensor([prompt_ids], device=device, dtype=torch.long)
                with torch.no_grad():
                    gen_ids = model.generate(
                        prompt_ids=p_tensor,
                        max_new_tokens=100,
                        temperature=0.7,
                        top_k=40,
                        top_p=0.9,
                    )
                out_tokens = gen_ids[0, len(prompt_ids):].tolist()
                gen_text = sp.decode_ids(out_tokens)
                with open(sample_file, "w", encoding="utf-8") as f:
                    f.write(f"--- Prompt ---\n{test_prompt}\n\n--- Generated ---\n{gen_text}\n")
                print(f"Sample generated at {sample_file}")
                model.train()

        print("\nTraining completed successfully.")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    finally:
        val_gen.stop()
        for g in train_gens:
            g.stop()
        if writer:
            writer.close()
        if use_wandb:
            wandb.finish()


if __name__ == "__main__":
    main()
