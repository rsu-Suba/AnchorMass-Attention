from dataclasses import dataclass, fields
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        head_dim: int,
        max_position_embeddings: int = 262144,
        base: float = 10000.0,
        scaling_type: Optional[str] = "dynamic_ntk",
        original_seq_len: int = 1024,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.base = float(base)
        self.scaling_type = scaling_type
        self.original_seq_len = int(original_seq_len)
        self.max_position_embeddings = int(max_position_embeddings)

        inv_freq = 1.0 / (self.base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        initial_len = min(self.max_position_embeddings, 4096)
        t = torch.arange(initial_len, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq.float())
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
        self.session_len = None

    def _compute_inv_freq_for_len(self, seq_len: int, device: torch.device):
        if self.scaling_type == "dynamic_ntk" and seq_len > self.original_seq_len:
            scale = float(seq_len) / float(self.original_seq_len)
            new_base = self.base * (scale ** (self.head_dim / (self.head_dim - 2)))
            return 1.0 / (new_base ** (torch.arange(0, self.head_dim, 2, device=device, dtype=torch.float32) / self.head_dim))
        elif self.scaling_type == "linear" and seq_len > self.original_seq_len:
            scale = float(seq_len) / float(self.original_seq_len)
            return (1.0 / (self.base ** (torch.arange(0, self.head_dim, 2, device=device, dtype=torch.float32) / self.head_dim))) / scale
        return self.inv_freq.to(device=device, dtype=torch.float32)

    def set_session_length(self, total_seq_len: int, device: Optional[torch.device] = None):
        dev = device if device is not None else self.cos_cached.device
        target_len = max(total_seq_len + 128, 4096)
        with torch.amp.autocast(device_type="cuda", enabled=False):
            inv_f = self._compute_inv_freq_for_len(total_seq_len, dev).float()
            t = torch.arange(target_len, device=dev, dtype=torch.float32)
            freqs = torch.outer(t, inv_f)
            emb = torch.cat([freqs, freqs], dim=-1)
            self.cos_cached = emb.cos()
            self.sin_cached = emb.sin()
        self.session_len = total_seq_len

    def reset_session(self):
        dev = self.cos_cached.device
        initial_len = min(self.max_position_embeddings, 4096)
        with torch.amp.autocast(device_type="cuda", enabled=False):
            t = torch.arange(initial_len, device=dev, dtype=torch.float32)
            freqs = torch.outer(t, self.inv_freq.to(device=dev, dtype=torch.float32))
            emb = torch.cat([freqs, freqs], dim=-1)
            self.cos_cached = emb.cos()
            self.sin_cached = emb.sin()
        self.session_len = None

    def ensure_len(self, required_len: int, device: torch.device):
        if required_len > self.cos_cached.shape[0]:
            self.set_session_length(required_len, device=device)


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, dropout: float = 0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        self.num_rep = num_heads // num_kv_heads

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, num_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.dropout_p = dropout
        self.rope = RotaryEmbedding(self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[tuple] = None,
        offset: int = 0,
        max_cache_len: Optional[int] = None,
        sink_tokens: int = 16,
        eviction_policy: str = "fifo",
        use_cache: bool = False,
    ):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim)

        self.rope.ensure_len(offset + T, device=x.device)
        seq_pos = torch.arange(offset, offset + T, device=x.device)
        cos = self.rope.cos_cached[seq_pos].unsqueeze(0).unsqueeze(2).to(dtype=q.dtype)
        sin = self.rope.sin_cached[seq_pos].unsqueeze(0).unsqueeze(2).to(dtype=q.dtype)
        q = (q * cos) + (rotate_half(q) * sin)
        k = (k * cos) + (rotate_half(k) * sin)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if kv_cache is not None:
            k = torch.cat([kv_cache[0], k], dim=2)
            v = torch.cat([kv_cache[1], v], dim=2)
            if max_cache_len is not None and k.shape[2] > max_cache_len:
                eff_sink = min(sink_tokens, max_cache_len // 2)
                over = k.shape[2] - max_cache_len

                if eviction_policy == "priority":
                    recent_protect = max(1, max_cache_len // 2)
                    mid_start = eff_sink
                    mid_end = k.shape[2] - recent_protect
                    mid_len = mid_end - mid_start

                    if mid_len > over:
                        mid_v = v[:, :, mid_start:mid_end, :]
                        scores = mid_v.norm(dim=-1).mean(dim=(0, 1))
                        keep_count = mid_len - over
                        _, top_local = torch.topk(scores, k=keep_count, largest=True, sorted=False)
                        top_local, _ = torch.sort(top_local)
                        actual_mid = top_local + mid_start

                        keep_all = torch.cat([
                            torch.arange(0, eff_sink, device=k.device),
                            actual_mid,
                            torch.arange(mid_end, k.shape[2], device=k.device),
                        ])
                        k = k[:, :, keep_all]
                        v = v[:, :, keep_all]
                    else:
                        window_size = max_cache_len - eff_sink
                        k = torch.cat([k[:, :, :eff_sink], k[:, :, -window_size:]], dim=2)
                        v = torch.cat([v[:, :, :eff_sink], v[:, :, -window_size:]], dim=2)
                else:
                    window_size = max_cache_len - eff_sink
                    k = torch.cat([k[:, :, :eff_sink], k[:, :, -window_size:]], dim=2)
                    v = torch.cat([v[:, :, :eff_sink], v[:, :, -window_size:]], dim=2)
        new_cache = (k, v) if (use_cache or kv_cache is not None or not self.training) else None

        if self.num_rep > 1:
            k = k.repeat_interleave(self.num_rep, dim=1)
            v = v.repeat_interleave(self.num_rep, dim=1)

        out = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=(kv_cache is None and T > 1),
            dropout_p=self.dropout_p if self.training else 0.0,
        )
        out = out.transpose(1, 2).contiguous().view(B, T, self.dim)
        return self.out_proj(out), new_cache


@dataclass
class GPTConfig:
    vocab_size: int = 28000
    dim: int = 1024
    ff_dim: int = 4096
    ff_dims: Optional[list[int]] = None
    num_heads: int = 16
    num_kv_heads: int = 4
    num_layers: int = 32
    seq_len: int = 512
    dropout: float = 0.0
    gradient_checkpointing: bool = False
    tie_weights: bool = False


class GPTBlock(nn.Module):
    def __init__(self, cfg: GPTConfig, ff_dim: Optional[int] = None):
        super().__init__()
        self.norm1 = nn.RMSNorm(cfg.dim)
        self.attn = Attention(cfg.dim, cfg.num_heads, cfg.num_kv_heads, dropout=cfg.dropout)
        self.norm2 = nn.RMSNorm(cfg.dim)
        actual_ff_dim = ff_dim if ff_dim is not None else cfg.ff_dim
        self.ffn = SwiGLU(cfg.dim, actual_ff_dim)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[tuple] = None,
        offset: int = 0,
        max_cache_len: Optional[int] = None,
        sink_tokens: int = 16,
        eviction_policy: str = "fifo",
        use_cache: bool = False,
    ):
        attn_out, new_cache = self.attn(
            self.norm1(x),
            kv_cache=kv_cache,
            offset=offset,
            max_cache_len=max_cache_len,
            sink_tokens=sink_tokens,
            eviction_policy=eviction_policy,
            use_cache=use_cache,
        )
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x, new_cache


class GPTLanguageModel(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.dim)
        if cfg.ff_dims is not None:
            assert len(cfg.ff_dims) == cfg.num_layers, f"len(ff_dims)={len(cfg.ff_dims)} != num_layers={cfg.num_layers}"
            layer_ff_dims = cfg.ff_dims
        else:
            layer_ff_dims = [cfg.ff_dim] * cfg.num_layers

        self.blocks = nn.ModuleList([GPTBlock(cfg, ff_dim=layer_ff_dims[i]) for i in range(cfg.num_layers)])
        self.norm = nn.RMSNorm(cfg.dim)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        if self.cfg.tie_weights:
            self.lm_head.weight = self.embed.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02 / (2 * self.cfg.num_layers) ** 0.5)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def set_session_length(self, total_seq_len: int, device: Optional[torch.device] = None):
        for block in self.blocks:
            block.attn.rope.set_session_length(total_seq_len, device=device)

    def reset_session(self):
        for block in self.blocks:
            block.attn.rope.reset_session()

    def forward(
        self,
        input_ids: torch.Tensor,
        kv_caches: Optional[list] = None,
        use_cache: bool = False,
        offset: int = 0,
        max_cache_len: Optional[int] = None,
        sink_tokens: int = 16,
        eviction_policy: str = "fifo",
    ):
        x = self.embed(input_ids)
        need_cache = use_cache or (kv_caches is not None)
        new_caches = [] if need_cache else None

        for i, block in enumerate(self.blocks):
            cache = kv_caches[i] if kv_caches is not None else None
            if self.cfg.gradient_checkpointing and self.training and not need_cache:
                x, _ = checkpoint(block, x, None, 0, use_reentrant=False)
            else:
                x, new_cache = block(
                    x,
                    kv_cache=cache,
                    offset=offset,
                    max_cache_len=max_cache_len,
                    sink_tokens=sink_tokens,
                    eviction_policy=eviction_policy,
                    use_cache=need_cache,
                )
                if need_cache:
                    new_caches.append(new_cache)
        x = self.norm(x)
        logits = self.lm_head(x)
        return (logits, new_caches) if need_cache else logits

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 100,
        min_new_tokens: int = 0,
        temperature: float = 0.8,
        top_k: int = 40,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        eos_id: Optional[int] = 3,
    ) -> torch.Tensor:
        self.eval()
        B, prompt_len = prompt_ids.shape
        total_len = prompt_len + max_new_tokens
        if total_len > self.cfg.seq_len:
            self.set_session_length(total_len, device=prompt_ids.device)
        idx_cond = prompt_ids

        logits, kv_caches = self.forward(idx_cond, use_cache=True, offset=0)
        logits = logits[:, -1, :]

        curr_ids = prompt_ids.clone()
        offset = idx_cond.shape[1]

        for step in range(max_new_tokens):
            if step < min_new_tokens and eos_id is not None:
                logits[:, eos_id] = -1e9

            if repetition_penalty != 1.0:
                for b in range(B):
                    used = torch.unique(curr_ids[b])
                    logits[b, used] = torch.where(
                        logits[b, used] > 0,
                        logits[b, used] / repetition_penalty,
                        logits[b, used] * repetition_penalty,
                    )

            if temperature == 0.0:
                next_id = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("Inf")
                if 0.0 < top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    probs = F.softmax(sorted_logits, dim=-1)
                    sorted_logits[(torch.cumsum(probs, dim=-1) - probs) > top_p] = -float("Inf")
                    logits.scatter_(1, sorted_indices, sorted_logits)
                next_id = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)

            curr_ids = torch.cat([curr_ids, next_id], dim=1)
            if eos_id is not None and (next_id == eos_id).all() and step >= min_new_tokens:
                break

            logits, kv_caches = self.forward(next_id, kv_caches=kv_caches, offset=offset)
            logits = logits[:, -1, :]
            offset += 1

        return curr_ids


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def build_model(params: dict) -> GPTLanguageModel:
    valid_keys = {f.name for f in fields(GPTConfig)}
    cfg = GPTConfig(**{k.lower(): v for k, v in params.items() if k.lower() in valid_keys})
    return GPTLanguageModel(cfg)


if __name__ == "__main__":
    cfg = GPTConfig(vocab_size=100, dim=64, ff_dim=128, num_heads=4, num_kv_heads=2, num_layers=2, seq_len=32)
    m = GPTLanguageModel(cfg)
    x = torch.randint(0, 100, (2, 16))
    out = m(x)
    assert out.shape == (2, 16, 100), f"Expected (2, 16, 100), got {out.shape}"
    gen = m.generate(x[:, :4], max_new_tokens=5)
    assert gen.shape == (2, 9), f"Expected (2, 9), got {gen.shape}"
    print("Smoke test passed: model_new.py is working as expected.")
