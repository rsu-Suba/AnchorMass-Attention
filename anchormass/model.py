import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention, create_block_mask


@dataclass
class AnchorMassConfig:
    vocab_size: int = 28000
    dim: int = 512
    ff_dim: int = 2048
    num_heads: int = 16
    num_kv_heads: int = 4
    num_layers: int = 24
    seq_len: int = 1024
    chunk_size: int = 64
    sink_tokens: int = 16
    stride: int = 4
    mem_tokens: int = 4
    eos_id: Optional[int] = 3
    use_tome: bool = True
    max_global_cache: int = 256
    merge_batch_size: int = 16
    sim_threshold: float = 0.75
    outlier_frac: float = 0.125
    comp_per_block: int = 8
    use_flex: bool = False
    hybrid_train_prob: float = 0.25
    train_chunk_size: int = 2048
    rope_base: float = 10000.0
    rope_scaling_type: Optional[str] = "dynamic_ntk"
    original_seq_len: int = 1024
    max_position_embeddings: int = 262144
    eviction_policy: str = "adaptive"  # "adaptive", "merge", "priority", "uniform_merge", "fifo"

    @classmethod
    def from_dict(cls, d: dict):
        return cls(
            vocab_size=d.get("VOCAB_SIZE", 28000),
            dim=d.get("DIM", 512),
            ff_dim=d.get("FF_DIM", 2048),
            num_heads=d.get("NUM_HEADS", 16),
            num_kv_heads=d.get("NUM_KV_HEADS", 4),
            num_layers=d.get("NUM_LAYERS", 24),
            seq_len=d.get("SEQ_LEN", 512),
            chunk_size=d.get("CHUNK_SIZE", 64),
            sink_tokens=d.get("SINK_TOKENS", 16),
            stride=d.get("STRIDE", 4),
            mem_tokens=d.get("MEM_TOKENS", 4),
            eos_id=d.get("EOS_ID", 3),
            use_tome=d.get("USE_TOME", True),
            max_global_cache=d.get("MAX_GLOBAL_CACHE", 256),
            merge_batch_size=d.get("MERGE_BATCH_SIZE", 16),
            sim_threshold=d.get("SIM_THRESHOLD", 0.75),
            outlier_frac=d.get("OUTLIER_FRAC", 0.125),
            comp_per_block=d.get("COMP_PER_BLOCK", 8),
            use_flex=d.get("USE_FLEX", False),
            hybrid_train_prob=d.get("HYBRID_TRAIN_PROB", 0.25),
            train_chunk_size=d.get("TRAIN_CHUNK_SIZE", 2048),
            rope_base=d.get("ROPE_BASE", 10000.0),
            rope_scaling_type=d.get("ROPE_SCALING_TYPE", "dynamic_ntk"),
            original_seq_len=d.get("ORIGINAL_SEQ_LEN", 1024),
            max_position_embeddings=d.get("MAX_POSITION_EMBEDDINGS", 262144),
            eviction_policy=d.get("EVICTION_POLICY", "adaptive"),
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 262144,
        base: float = 10000.0,
        scaling_type: Optional[str] = "dynamic_ntk",
        original_seq_len: int = 1024,
    ):
        super().__init__()
        self.dim = dim
        self.base = float(base)
        self.scaling_type = scaling_type
        self.original_seq_len = int(original_seq_len)
        self.max_seq_len_cached = int(max_position_embeddings)

        inv_freq = 1.0 / (self.base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        initial_len = min(self.max_seq_len_cached, 4096)
        self._build_cache(initial_len, inv_freq.device, inv_freq)
        self.session_len: Optional[int] = None

    def _compute_inv_freq_for_len(self, seq_len: int, device: torch.device) -> torch.Tensor:
        if self.scaling_type == "dynamic_ntk" and seq_len > self.original_seq_len:
            scale = float(seq_len) / float(self.original_seq_len)
            new_base = self.base * (scale ** (self.dim / (self.dim - 2)))
            return 1.0 / (new_base ** (torch.arange(0, self.dim, 2, device=device).float() / self.dim))
        if self.scaling_type == "linear" and seq_len > self.original_seq_len:
            scale = float(seq_len) / float(self.original_seq_len)
            return (1.0 / (self.base ** (torch.arange(0, self.dim, 2, device=device).float() / self.dim))) / scale
        return self.inv_freq.to(device)

    def _build_cache(self, target_len: int, device: torch.device, inv_f: torch.Tensor):
        t = torch.arange(target_len, device=device, dtype=inv_f.dtype)
        freqs = torch.outer(t, inv_f)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)
        self.cached_len = target_len

    def set_session_length(self, total_seq_len: int, device: Optional[torch.device] = None):
        dev = device if device is not None else self.cos_cached.device
        if self.session_len == total_seq_len and self.cached_len >= total_seq_len and self.cos_cached.device == dev:
            return
        target_len = max(total_seq_len + 128, 4096)
        inv_f = self._compute_inv_freq_for_len(total_seq_len, dev)
        self._build_cache(target_len, dev, inv_f)
        self.session_len = total_seq_len

    def reset_session(self):
        if self.session_len is None and self.cached_len == 4096:
            return
        dev = self.cos_cached.device
        self._build_cache(4096, dev, self.inv_freq.to(dev))
        self.session_len = None

    def forward(self, x: torch.Tensor, offset: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        T = x.shape[1]
        seq_len = T + offset
        if seq_len > self.cached_len or self.cos_cached.device != x.device:
            dev = x.device
            target_len = max(seq_len + 1024, self.cached_len * 2)
            scale_ref = max(seq_len, self.session_len or 0)
            inv_f = self._compute_inv_freq_for_len(scale_ref, dev)
            self._build_cache(target_len, dev, inv_f)

        return (
            self.cos_cached[:, :, offset : offset + T, :].to(x.dtype),
            self.sin_cached[:, :, offset : offset + T, :].to(x.dtype),
        )


@lru_cache(maxsize=32)
def _cached_stride_indices(seq_len: int, sink: int, stride: int) -> torch.Tensor:
    indices = list(range(min(sink, seq_len)))
    if seq_len > sink:
        s = stride
        pos = sink
        nxt = max(sink * 2, sink + 1)
        while pos < seq_len:
            indices.append(pos)
            pos += s
            if pos >= nxt:
                s *= 2
                nxt *= 2
    return torch.tensor(indices, dtype=torch.long)


class AnchorMassAttention(nn.Module):
    def __init__(self, cfg: AnchorMassConfig):
        super().__init__()
        self.dim = cfg.dim
        self.num_heads = cfg.num_heads
        self.num_kv_heads = cfg.num_kv_heads
        self.head_dim = self.dim // self.num_heads
        self.chunk_size = getattr(cfg, "chunk_size", 64)
        self.sink_tokens = getattr(cfg, "sink_tokens", 16)
        self.stride = getattr(cfg, "stride", 4)

        self.local_heads = self.num_heads // 2
        self.global_heads = self.num_heads - self.local_heads
        self.local_kv_heads = max(1, self.num_kv_heads // 2)
        self.global_kv_heads = max(1, self.num_kv_heads - self.local_kv_heads)

        self.q_proj = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.k_proj = nn.Linear(cfg.dim, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.dim, self.num_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(cfg.dim, cfg.dim, bias=False)

        self.rope = RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=getattr(cfg, "max_position_embeddings", 262144),
            base=getattr(cfg, "rope_base", 10000.0),
            scaling_type=getattr(cfg, "rope_scaling_type", "dynamic_ntk"),
            original_seq_len=getattr(cfg, "original_seq_len", 1024),
        )

        self.use_tome = getattr(cfg, "use_tome", True)
        self.max_global_cache = getattr(cfg, "max_global_cache", 256)
        self.eviction_policy = getattr(cfg, "eviction_policy", "adaptive")
        self.merge_batch_size = max(1, getattr(cfg, "merge_batch_size", 16))
        self.sim_threshold = getattr(cfg, "sim_threshold", 0.75)
        self.outlier_frac = getattr(cfg, "outlier_frac", 0.125)
        self.comp_per_block = max(1, getattr(cfg, "comp_per_block", 8))
        self.use_flex = bool(getattr(cfg, "use_flex", False))
        self.hybrid_train_prob = float(getattr(cfg, "hybrid_train_prob", 0.25))

        self._mask_cache: dict = {}

    def _active_compress_mode(self) -> str:
        if self.eviction_policy == "priority":
            return "priority"
        if self.training and self.hybrid_train_prob > 0.0:
            if torch.rand((), device=self.q_proj.weight.device) < self.hybrid_train_prob:
                return "priority"
        return "merge"

    def _selective_block_compress(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        n: Optional[torch.Tensor] = None,
        mode: Optional[str] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, H, L, D = k.shape
        dev = k.device
        G = self.comp_per_block
        S = 2 * G
        active_mode = mode or self._active_compress_mode()

        if n is None:
            n = torch.ones((B, H, L), device=dev, dtype=torch.float32)
        else:
            n = n.to(torch.float32)
            if n.dim() == 4:
                n = n.squeeze(1)

        if L <= S:
            pad = S - L
            pos = torch.arange(L, device=dev, dtype=torch.long)
            if pad > 0:
                k = F.pad(k, (0, 0, 0, pad))
                v = F.pad(v, (0, 0, 0, pad))
                n = F.pad(n, (0, pad))
                pos = F.pad(pos, (0, pad), value=max(0, L - 1))
            out_pos = pos.unsqueeze(0).expand(B, S).contiguous()
            return k.contiguous(), v.contiguous(), n.contiguous(), out_pos

        if active_mode == "priority":
            with torch.no_grad():
                valid = n > 0
                valid_b = valid.any(dim=1)  # [B, L]
                w = n * valid.float()
                v_bar = (v * w.unsqueeze(-1)).sum(dim=-2, keepdim=True) / w.sum(dim=-1, keepdim=True).clamp_min(1e-6).unsqueeze(-1)
                vn = F.normalize(v, dim=-1, eps=1e-6)
                bn = F.normalize(v_bar, dim=-1, eps=1e-6)
                dist = 1.0 - (vn * bn).sum(dim=-1).clamp(-1.0, 1.0)
                score_h = v.norm(dim=-1) * (dist + 0.1) * torch.log1p(n.clamp_min(0.0))
                score_b = torch.where(
                    valid_b,
                    score_h.mean(dim=1),
                    torch.full((B, L), -1e9, device=dev, dtype=torch.float32),
                )
                _, top_idx = torch.topk(score_b, k=S, dim=-1, largest=True, sorted=False)
                top_idx, _ = torch.sort(top_idx, dim=-1)  # [B, S]

            idx_h = top_idx.unsqueeze(1).expand(B, H, S)
            idx_d = idx_h.unsqueeze(-1).expand(B, H, S, D)
            out_k = k.gather(2, idx_d).contiguous()
            out_v = v.gather(2, idx_d).contiguous()
            out_n = n.gather(2, idx_h).contiguous()
            return out_k, out_v, out_n, top_idx.contiguous()

        gsize = math.ceil(L / G)
        total = G * gsize
        if total > L:
            k = F.pad(k, (0, 0, 0, total - L))
            v = F.pad(v, (0, 0, 0, total - L))
            n = F.pad(n, (0, total - L))

        kg = k.view(B, H, G, gsize, D)
        vg = v.view(B, H, G, gsize, D)
        ng = n.view(B, H, G, gsize)
        valid = ng > 0
        vf = valid.float()

        with torch.no_grad():
            w_init = ng * vf
            v_init_mean = (vg * w_init.unsqueeze(-1)).sum(-2) / w_init.sum(-1, keepdim=True).clamp_min(1e-6)
            vn = F.normalize(vg, dim=-1, eps=1e-6)
            mn = F.normalize(v_init_mean, dim=-1, eps=1e-6).unsqueeze(-2)
            sim = torch.where(valid, (vn * mn).sum(-1), torch.ones_like(ng))
            minsim, minidx = sim.mean(dim=1).min(dim=-1)  # [B, G]

            vnorm = vg.norm(dim=-1)
            meannorm = (vnorm * vf).sum(-1, keepdim=True) / vf.sum(-1, keepdim=True).clamp_min(1e-6)
            rel = torch.where(valid, vnorm / (meannorm + 1e-6), torch.zeros_like(vnorm))
            maxrel = rel.mean(dim=1).amax(dim=-1)  # [B, G]

            valid_count = valid.any(dim=1).sum(dim=-1)  # [B, G]
            emit = ((minsim < self.sim_threshold) | (maxrel > 2.0)) & (valid_count > 1)

            minidx_h = minidx.unsqueeze(1).expand(B, H, G)
            is_outlier = torch.zeros_like(valid).scatter(
                -1, minidx_h.unsqueeze(-1), emit.unsqueeze(1).unsqueeze(-1).expand(B, H, G, 1)
            )
            bg_valid = valid & (~is_outlier)
            bg_vf = bg_valid.float()
            pos_1based = torch.arange(1, gsize + 1, device=dev, dtype=torch.float32)
            last_bg = (pos_1based * bg_vf).argmax(dim=-1).long()  # [B, H, G]
            last_bg_b = (pos_1based * bg_valid.any(dim=1).float()).argmax(dim=-1).long()  # [B, G]

        w_bg = ng * bg_vf
        vmean = (vg * w_bg.unsqueeze(-1)).sum(-2) / w_bg.sum(-1, keepdim=True).clamp_min(1e-6)
        nsum_bg = w_bg.sum(-1)

        k_anchor = kg.gather(-2, last_bg.unsqueeze(-1).unsqueeze(-1).expand(B, H, G, 1, D)).squeeze(-2)

        gidx_d = minidx_h.unsqueeze(-1).unsqueeze(-1).expand(B, H, G, 1, D)
        ef = emit.unsqueeze(1).unsqueeze(-1).float()
        k_extra = kg.gather(-2, gidx_d).squeeze(-2) * ef
        v_extra = vg.gather(-2, gidx_d).squeeze(-2) * ef
        n_extra = ng.gather(-1, minidx_h.unsqueeze(-1)).squeeze(-1) * ef.squeeze(-1)

        g_base = (torch.arange(G, device=dev, dtype=torch.long) * gsize).unsqueeze(0)
        pos_bg = (g_base + last_bg_b).clamp(max=L - 1)
        pos_ex = (g_base + minidx).clamp(max=L - 1)
        out_pos = torch.stack([pos_bg, pos_ex], dim=-1).view(B, S).contiguous()

        out_k = torch.stack([k_anchor.to(k.dtype), k_extra.to(k.dtype)], dim=-2).view(B, H, S, D)
        out_v = torch.stack([vmean.to(v.dtype), v_extra.to(v.dtype)], dim=-2).view(B, H, S, D)
        out_n = torch.stack([nsum_bg, n_extra], dim=-1).view(B, H, S)
        return out_k.contiguous(), out_v.contiguous(), out_n.contiguous(), out_pos

    def _compress_stream(
        self,
        k_raw: torch.Tensor,
        v_raw: torch.Tensor,
        n_raw: Optional[torch.Tensor] = None,
        compress_tail: bool = True,
        mode: Optional[str] = None,
    ):
        B, H, L, D = k_raw.shape
        dev = k_raw.device
        C = self.chunk_size
        S = 2 * self.comp_per_block
        active_mode = mode or self._active_compress_mode()

        if n_raw is None:
            n_raw = torch.ones((B, H, L), device=dev, dtype=torch.float32)

        n_full = L // C
        rem = L % C
        sum_k_list, sum_v_list, sum_n_list, rep_list = [], [], [], []

        if n_full > 0:
            full_len = n_full * C
            fk = k_raw[:, :, :full_len].view(B, H, n_full, C, D).permute(0, 2, 1, 3, 4).reshape(B * n_full, H, C, D)
            fv = v_raw[:, :, :full_len].view(B, H, n_full, C, D).permute(0, 2, 1, 3, 4).reshape(B * n_full, H, C, D)
            fn = n_raw[:, :, :full_len].view(B, H, n_full, C).permute(0, 2, 1, 3).reshape(B * n_full, H, C)
            ck, cv, cn, cpos = self._selective_block_compress(fk, fv, fn, mode=active_mode)

            sum_k_list.append(ck.view(B, n_full, H, S, D).permute(0, 2, 1, 3, 4).reshape(B, H, n_full * S, D))
            sum_v_list.append(cv.view(B, n_full, H, S, D).permute(0, 2, 1, 3, 4).reshape(B, H, n_full * S, D))
            sum_n_list.append(cn.view(B, n_full, H, S).permute(0, 2, 1, 3).reshape(B, H, n_full * S))

            b_off = (torch.arange(n_full, device=dev, dtype=torch.long) * C).view(1, n_full, 1)
            rep_list.append((cpos.view(B, n_full, S) + b_off).reshape(B, n_full * S))

        base_t = n_full * C
        if rem > 0 and compress_tail:
            ck1, cv1, cn1, cpos1 = self._selective_block_compress(
                k_raw[:, :, base_t:], v_raw[:, :, base_t:], n_raw[:, :, base_t:], mode=active_mode
            )
            sum_k_list.append(ck1)
            sum_v_list.append(cv1)
            sum_n_list.append(cn1)
            rep_list.append(cpos1 + base_t)
            pk = k_raw.new_zeros((B, H, 0, D))
            pv = v_raw.new_zeros((B, H, 0, D))
            pn = n_raw.new_zeros((B, H, 0))
            p_rep = torch.empty((B, 0), device=dev, dtype=torch.long)
        elif rem > 0:
            pk = k_raw[:, :, base_t:].contiguous()
            pv = v_raw[:, :, base_t:].contiguous()
            pn = n_raw[:, :, base_t:].contiguous()
            p_rep = torch.arange(base_t, L, device=dev, dtype=torch.long).unsqueeze(0).expand(B, rem)
        else:
            pk = k_raw.new_zeros((B, H, 0, D))
            pv = v_raw.new_zeros((B, H, 0, D))
            pn = n_raw.new_zeros((B, H, 0))
            p_rep = torch.empty((B, 0), device=dev, dtype=torch.long)

        if sum_k_list:
            sum_k = torch.cat(sum_k_list, dim=2)
            sum_v = torch.cat(sum_v_list, dim=2)
            sum_n = torch.cat(sum_n_list, dim=2)
            sum_rep = torch.cat(rep_list, dim=1)
        else:
            sum_k = k_raw.new_zeros((B, H, 0, D))
            sum_v = v_raw.new_zeros((B, H, 0, D))
            sum_n = n_raw.new_zeros((B, H, 0))
            sum_rep = torch.empty((B, 0), device=dev, dtype=torch.long)

        return sum_k, sum_v, sum_n, sum_rep, (pk, pv, pn), p_rep

    def _prefill_compress(self, k_g_raw: torch.Tensor, v_g_raw: torch.Tensor, is_cache: bool = False):
        B, H, T, D = k_g_raw.shape
        dev = k_g_raw.device
        sink = min(self.sink_tokens, T)
        k_sink = k_g_raw[:, :, :sink]
        v_sink = v_g_raw[:, :, :sink]
        n_sink = torch.ones((B, H, sink), device=dev, dtype=torch.float32)
        rep_sink = torch.arange(sink, device=dev, dtype=torch.long).unsqueeze(0).expand(B, sink)
        if T <= sink:
            return k_sink, v_sink, n_sink, rep_sink

        mode = self._active_compress_mode()
        sum_k, sum_v, sum_n, sum_rep, _, _ = self._compress_stream(
            k_g_raw[:, :, sink:], v_g_raw[:, :, sink:], compress_tail=True, mode=mode
        )
        k_all = torch.cat([k_sink, sum_k], dim=2)
        v_all = torch.cat([v_sink, sum_v], dim=2)
        n_all = torch.cat([n_sink, sum_n], dim=2)
        rep_pos = torch.cat([rep_sink, sum_rep + sink], dim=1)

        if is_cache:
            keep = n_all.sum(dim=(0, 1)) > 0
            if not bool(keep.all()):
                k_all, v_all, n_all = k_all[:, :, keep], v_all[:, :, keep], n_all[:, :, keep]
                rep_pos = rep_pos[:, keep]
            pre_len = k_all.shape[2]
            k_all, v_all, n_all = self._enforce_budget(k_all, v_all, n_all, sink, policy_override=mode)
            if k_all.shape[2] < pre_len:
                dropped = pre_len - k_all.shape[2]
                rep_pos = torch.cat([rep_pos[:, :sink], rep_pos[:, sink + dropped :]], dim=1)

        return k_all, v_all, n_all, rep_pos

    def _enforce_budget(
        self,
        k_all: torch.Tensor,
        v_all: torch.Tensor,
        n_all: torch.Tensor,
        sink: int,
        policy_override: Optional[str] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, H, L_total, D = k_all.shape
        over = L_total - (sink + self.max_global_cache)
        if over <= 0:
            return k_all, v_all, n_all

        dev = k_all.device
        policy = "priority" if policy_override == "priority" else self.eviction_policy

        if policy == "fifo":
            return (
                torch.cat([k_all[:, :, :sink], k_all[:, :, sink + over :]], dim=2),
                torch.cat([v_all[:, :, :sink], v_all[:, :, sink + over :]], dim=2),
                torch.cat([n_all[:, :, :sink], n_all[:, :, sink + over :]], dim=2),
            )

        if policy == "priority":
            total_cand = L_total - sink
            recent_protect = min(self.max_global_cache // 2, max(1, total_cand - over))
            mid_start = sink
            mid_end = L_total - recent_protect
            mid_len = mid_end - mid_start
            if mid_len <= over:
                return (
                    torch.cat([k_all[:, :, :sink], k_all[:, :, sink + over :]], dim=2),
                    torch.cat([v_all[:, :, :sink], v_all[:, :, sink + over :]], dim=2),
                    torch.cat([n_all[:, :, :sink], n_all[:, :, sink + over :]], dim=2),
                )
            mid_k = k_all[:, :, mid_start:mid_end, :]
            mid_v = v_all[:, :, mid_start:mid_end, :]
            mid_n = n_all[:, :, mid_start:mid_end]

            with torch.no_grad():
                valid_m = mid_n > 0
                scores_h = mid_v.norm(dim=-1) * torch.log1p(mid_n.clamp_min(0.0))
                scores_b = torch.where(
                    valid_m.any(dim=1),
                    scores_h.mean(dim=1),
                    torch.full((B, mid_len), -1e9, device=dev, dtype=torch.float32),
                )
                keep_count = mid_len - over
                _, top_local = torch.topk(scores_b, k=keep_count, dim=-1, largest=True, sorted=False)
                top_local, _ = torch.sort(top_local, dim=-1)

            idx_h = top_local.unsqueeze(1).expand(B, H, keep_count)
            idx_d = idx_h.unsqueeze(-1).expand(B, H, keep_count, D)
            kept_k = mid_k.gather(2, idx_d)
            kept_v = mid_v.gather(2, idx_d)
            kept_n = mid_n.gather(2, idx_h)

            return (
                torch.cat([k_all[:, :, :sink], kept_k, k_all[:, :, mid_end:]], dim=2),
                torch.cat([v_all[:, :, :sink], kept_v, v_all[:, :, mid_end:]], dim=2),
                torch.cat([n_all[:, :, :sink], kept_n, n_all[:, :, mid_end:]], dim=2),
            )

        is_adaptive = policy in ("adaptive", "adaptive_merge")
        is_uniform = policy == "uniform_merge"

        while over > 0:
            L_cur = k_all.shape[2]
            cand_len = L_cur - sink
            if cand_len < 2:
                break

            if is_adaptive:
                protect = min(self.max_global_cache // 4, max(0, cand_len - 2))
                active_len = max(2, cand_len - protect)
            else:
                active_len = cand_len

            num_pairs = active_len // 2
            if num_pairs == 0:
                break

            m = min(over, num_pairs, self.merge_batch_size)

            with torch.no_grad():
                if is_adaptive and m < num_pairs:
                    span = num_pairs * 2
                    cv = v_all[:, :, sink : sink + span, :]
                    cn = n_all[:, :, sink : sink + span]
                    slot_e = (cv.norm(dim=-1) * torch.log1p(cn.clamp_min(0.0))).mean(dim=1)  # [B, span]
                    pair_e = slot_e[:, 0::2] + slot_e[:, 1::2]  # [B, num_pairs]
                    _, pair_idx = torch.topk(pair_e, k=m, dim=-1, largest=False, sorted=True)  # [B, m]
                else:
                    pair_idx = torch.arange(m, device=dev, dtype=torch.long).unsqueeze(0).expand(B, m)

                idx_a = sink + pair_idx * 2  # [B, m]
                idx_b = idx_a + 1  # [B, m]

            idx_a_h = idx_a.unsqueeze(1).expand(B, H, m)
            idx_b_h = idx_b.unsqueeze(1).expand(B, H, m)
            idx_a_d = idx_a_h.unsqueeze(-1).expand(B, H, m, D)
            idx_b_d = idx_b_h.unsqueeze(-1).expand(B, H, m, D)

            ka, kb = k_all.gather(2, idx_a_d), k_all.gather(2, idx_b_d)
            va, vb = v_all.gather(2, idx_a_d), v_all.gather(2, idx_b_d)
            na, nb = n_all.gather(2, idx_a_h), n_all.gather(2, idx_b_h)

            with torch.no_grad():
                if is_uniform:
                    sa, sb = na.clamp_min(0.0), nb.clamp_min(0.0)
                else:
                    sa = va.norm(dim=-1) * torch.log1p(na.clamp_min(0.0))
                    sb = vb.norm(dim=-1) * torch.log1p(nb.clamp_min(0.0))

                w = sb / (sa + sb + 1e-6)
                only_a = (na > 0) & (nb <= 0)
                only_b = (nb > 0) & (na <= 0)
                w = torch.where(only_a, torch.zeros_like(w), torch.where(only_b, torch.ones_like(w), w))
                w_u = w.unsqueeze(-1)

            v_merged = ((1.0 - w_u) * va + w_u * vb).to(v_all.dtype)
            k_merged = torch.where(w_u >= 0.5, kb, ka)
            n_merged = na + nb

            k_all = k_all.scatter(2, idx_b_d, k_merged)
            v_all = v_all.scatter(2, idx_b_d, v_merged)
            n_all = n_all.scatter(2, idx_b_h, n_merged)

            keep_mask = torch.ones((B, L_cur), device=dev, dtype=torch.bool).scatter(1, idx_a, False)
            keep_idx = (
                torch.arange(L_cur, device=dev, dtype=torch.long)
                .unsqueeze(0)
                .expand(B, L_cur)[keep_mask]
                .view(B, L_cur - m)
            )
            keep_idx_h = keep_idx.unsqueeze(1).expand(B, H, L_cur - m)
            keep_idx_d = keep_idx_h.unsqueeze(-1).expand(B, H, L_cur - m, D)

            k_all = k_all.gather(2, keep_idx_d)
            v_all = v_all.gather(2, keep_idx_d)
            n_all = n_all.gather(2, keep_idx_h)
            over -= m

        return k_all.contiguous(), v_all.contiguous(), n_all.contiguous()

    def _flex_sliding_local(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        if not (self.use_flex and q.is_cuda):
            return None
        try:
            B, H, T, _ = q.shape
            C = self.chunk_size
            key = ("flex", B, H, T, C, str(q.device))
            bm = self._mask_cache.get(key)
            if bm is None:
                def mask_mod(b, h, qi, kj):
                    return (qi >= kj) & ((qi - kj) < C)
                bm = create_block_mask(mask_mod, B, H, T, T, device=q.device)
                if len(self._mask_cache) < 16:
                    self._mask_cache[key] = bm
            return flex_attention(q, k, v, block_mask=bm)
        except Exception:
            return None

    def _local_attention(
        self,
        q_loc_t: torch.Tensor,
        k_loc_t: torch.Tensor,
        v_loc_t: torch.Tensor,
        k_loc_c: Optional[torch.Tensor] = None,
        v_loc_c: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, H, T, D = q_loc_t.shape
        C = self.chunk_size
        L_c = k_loc_c.shape[2] if k_loc_c is not None else 0

        if L_c == 0 and T == 1:
            return F.scaled_dot_product_attention(q_loc_t, k_loc_t, v_loc_t, is_causal=False)
        if L_c == 0 and T <= C:
            return F.scaled_dot_product_attention(q_loc_t, k_loc_t, v_loc_t, is_causal=True)
        if L_c == 0:
            flex_out = self._flex_sliding_local(q_loc_t, k_loc_t, v_loc_t)
            if flex_out is not None:
                return flex_out

        pad_r = (C - (T % C)) % C
        T_pad = T + pad_r
        N = T_pad // C
        pad_l = C - L_c

        q_pad = F.pad(q_loc_t, (0, 0, 0, pad_r)) if pad_r > 0 else q_loc_t
        if L_c > 0:
            k_cat = torch.cat([k_loc_c, k_loc_t], dim=2)
            v_cat = torch.cat([v_loc_c, v_loc_t], dim=2)
        else:
            k_cat, v_cat = k_loc_t, v_loc_t

        k_pad = F.pad(k_cat, (0, 0, pad_l, pad_r))
        v_pad = F.pad(v_cat, (0, 0, pad_l, pad_r))

        k_blk = k_pad.view(B, H, N + 1, C, D)
        v_blk = v_pad.view(B, H, N + 1, C, D)
        k_win = torch.cat([k_blk[:, :, :-1], k_blk[:, :, 1:]], dim=3).permute(0, 2, 1, 3, 4).reshape(B * N, H, 2 * C, D)
        v_win = torch.cat([v_blk[:, :, :-1], v_blk[:, :, 1:]], dim=3).permute(0, 2, 1, 3, 4).reshape(B * N, H, 2 * C, D)
        q_win = q_pad.view(B, H, N, C, D).permute(0, 2, 1, 3, 4).reshape(B * N, H, C, D)

        mask_key = ("loc_win", N, C, pad_l, str(q_loc_t.device))
        win_mask = self._mask_cache.get(mask_key)
        if win_mask is None:
            qi = torch.arange(C, device=q_loc_t.device).unsqueeze(1)
            kj = torch.arange(2 * C, device=q_loc_t.device).unsqueeze(0)
            base_m = (kj <= C + qi) & (kj > qi)
            if pad_l == 0:
                win_mask = base_m.unsqueeze(0).unsqueeze(0)
            else:
                b_idx = torch.arange(N, device=q_loc_t.device).view(N, 1, 1, 1)
                valid_k = (b_idx * C + kj.view(1, 1, 1, 2 * C)) >= pad_l
                win_mask = base_m.view(1, 1, C, 2 * C) & valid_k
            if len(self._mask_cache) < 16:
                self._mask_cache[mask_key] = win_mask

        if win_mask.shape[0] == N and B > 1:
            attn_m = win_mask.unsqueeze(0).expand(B, N, 1, C, 2 * C).reshape(B * N, 1, C, 2 * C)
        else:
            attn_m = win_mask

        out_win = F.scaled_dot_product_attention(q_win, k_win, v_win, attn_mask=attn_m)
        out = out_win.view(B, N, H, C, D).permute(0, 2, 1, 3, 4).reshape(B, H, T_pad, D)
        return out[:, :, :T, :] if pad_r > 0 else out

    def _global_sdpa(
        self,
        q_glob_t: torch.Tensor,
        k_glob: torch.Tensor,
        v_glob: torch.Tensor,
        n_glob: Optional[torch.Tensor],
        rep_glob: int,
        causal_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, H_g, T_q, _ = q_glob_t.shape
        S = k_glob.shape[2]
        k_attn = k_glob.repeat_interleave(rep_glob, dim=1)
        v_attn = v_glob.repeat_interleave(rep_glob, dim=1)

        if n_glob is not None:
            n_exp = n_glob.repeat_interleave(rep_glob, dim=1).unsqueeze(2)
            log_n = torch.where(
                n_exp > 0,
                torch.log(n_exp.clamp_min(1e-6)).to(q_glob_t.dtype),
                torch.full_like(n_exp, float("-inf"), dtype=q_glob_t.dtype),
            )
        else:
            log_n = torch.zeros((B, H_g, 1, S), device=q_glob_t.device, dtype=q_glob_t.dtype)

        q_chunk = 4096 if T_q > 8192 else T_q
        outs = []
        for start in range(0, T_q, q_chunk):
            end = min(start + q_chunk, T_q)
            q_c = q_glob_t[:, :, start:end, :]
            if causal_mask is not None:
                cm_c = causal_mask[:, :, start:end, :]
                bias = torch.where(cm_c, log_n, float("-inf"))
            else:
                bias = log_n.expand(B, H_g, end - start, S)

            all_masked = ~torch.isfinite(bias).any(dim=-1, keepdim=True)
            if bool(all_masked.any()):
                bias = bias.clone()
                bias[..., :1] = torch.where(all_masked, torch.zeros_like(bias[..., :1]), bias[..., :1])

            outs.append(F.scaled_dot_product_attention(q_c, k_attn, v_attn, attn_mask=bias))

        return outs[0] if len(outs) == 1 else torch.cat(outs, dim=2)

    def _get_global_indices(self, seq_len: int, device: torch.device) -> torch.Tensor:
        return _cached_stride_indices(seq_len, self.sink_tokens, self.stride).to(device=device)

    def _build_global_causal_mask(self, q_len: int, global_indices: torch.Tensor, device: torch.device):
        key = ("glob_stride", q_len, tuple(global_indices.tolist()), str(device))
        hit = self._mask_cache.get(key)
        if hit is not None:
            return hit
        q_pos = torch.arange(q_len, device=device).unsqueeze(1)
        mask = (q_pos >= global_indices.unsqueeze(0)).unsqueeze(0).unsqueeze(0)
        if len(self._mask_cache) < 16:
            self._mask_cache[key] = mask
        return mask

    def forward(self, x: torch.Tensor, kv_cache=None, offset: int = 0, use_cache: bool = False):
        B, T, _ = x.shape
        dev = x.device
        C = self.chunk_size

        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim)

        cos, sin = self.rope(x, offset=offset)
        cos, sin = cos.permute(0, 2, 1, 3), sin.permute(0, 2, 1, 3)
        q = (q * cos) + (rotate_half(q) * sin)
        k = (k * cos) + (rotate_half(k) * sin)

        q_loc, q_glob = q.split([self.local_heads, self.global_heads], dim=2)
        k_loc, k_glob = k.split([self.local_kv_heads, self.global_kv_heads], dim=2)
        v_loc, v_glob = v.split([self.local_kv_heads, self.global_kv_heads], dim=2)

        rep_loc = self.local_heads // self.local_kv_heads
        rep_glob = self.global_heads // self.global_kv_heads

        q_loc_t = q_loc.transpose(1, 2)
        q_glob_t = q_glob.transpose(1, 2)
        k_loc_raw, v_loc_raw = k_loc.transpose(1, 2), v_loc.transpose(1, 2)
        k_glob_raw, v_glob_raw = k_glob.transpose(1, 2), v_glob.transpose(1, 2)

        has_cache = use_cache and kv_cache is not None and kv_cache[0] is not None
        if has_cache:
            loc_cache, glob_cache = kv_cache
            k_loc_c, v_loc_c = loc_cache
            if len(glob_cache) == 6:
                g_sink_k, g_sink_v, g_sum_k, g_sum_v, g_n, (pk, pv, pn) = glob_cache
            else:
                g_sink_k, g_sink_v = glob_cache
                g_sum_k = g_sum_v = g_n = None
                pk = g_sink_k.new_zeros((B, self.global_kv_heads, 0, self.head_dim))
                pv = g_sink_v.new_zeros((B, self.global_kv_heads, 0, self.head_dim))
                pn = torch.zeros((B, self.global_kv_heads, 0), device=dev, dtype=torch.float32)
        else:
            k_loc_c = v_loc_c = None
            g_sink_k = k_glob_raw.new_zeros((B, self.global_kv_heads, 0, self.head_dim))
            g_sink_v = v_glob_raw.new_zeros((B, self.global_kv_heads, 0, self.head_dim))
            g_sum_k = g_sum_v = g_n = None
            pk = k_glob_raw.new_zeros((B, self.global_kv_heads, 0, self.head_dim))
            pv = v_glob_raw.new_zeros((B, self.global_kv_heads, 0, self.head_dim))
            pn = torch.zeros((B, self.global_kv_heads, 0), device=dev, dtype=torch.float32)

        k_loc_t = k_loc_raw.repeat_interleave(rep_loc, dim=1)
        v_loc_t = v_loc_raw.repeat_interleave(rep_loc, dim=1)
        k_loc_c_exp = k_loc_c.repeat_interleave(rep_loc, dim=1) if k_loc_c is not None else None
        v_loc_c_exp = v_loc_c.repeat_interleave(rep_loc, dim=1) if v_loc_c is not None else None

        out_loc = self._local_attention(q_loc_t, k_loc_t, v_loc_t, k_loc_c_exp, v_loc_c_exp)

        if use_cache:
            k_loc_all = torch.cat([k_loc_c, k_loc_raw], dim=2) if k_loc_c is not None else k_loc_raw
            v_loc_all = torch.cat([v_loc_c, v_loc_raw], dim=2) if v_loc_c is not None else v_loc_raw
            new_loc_cache = (k_loc_all[:, :, -C:, :].contiguous(), v_loc_all[:, :, -C:, :].contiguous())
        else:
            new_loc_cache = None

        if not self.use_tome:
            k_full = torch.cat([g_sink_k, k_glob_raw], dim=2) if has_cache else k_glob_raw
            v_full = torch.cat([g_sink_v, v_glob_raw], dim=2) if has_cache else v_glob_raw
            g_idx = self._get_global_indices(k_full.shape[2], dev)
            k_sp, v_sp = k_full[:, :, g_idx, :], v_full[:, :, g_idx, :]
            c_mask = None if T == 1 else self._build_global_causal_mask(T, g_idx - offset, dev)
            out_glob = self._global_sdpa(q_glob_t, k_sp, v_sp, None, rep_glob, causal_mask=c_mask)
            new_glob_cache = (k_sp, v_sp, None, None, None, (pk, pv, pn)) if use_cache else None

        elif not use_cache:
            ck, cv, cn, rep_pos = self._prefill_compress(k_glob_raw, v_glob_raw, is_cache=False)
            q_pos = torch.arange(T, device=dev, dtype=torch.long).view(1, 1, T, 1)
            c_mask = q_pos >= rep_pos[:, : ck.shape[2]].view(B, 1, 1, -1)
            out_glob = self._global_sdpa(q_glob_t, ck, cv, cn, rep_glob, causal_mask=c_mask)
            new_glob_cache = None

        else:
            active_mode = self._active_compress_mode()
            sink_l = g_sink_k.shape[2]
            need_sink = max(0, min(self.sink_tokens - sink_l, T))
            if need_sink > 0:
                g_sink_k = torch.cat([g_sink_k, k_glob_raw[:, :, :need_sink]], dim=2)
                g_sink_v = torch.cat([g_sink_v, v_glob_raw[:, :, :need_sink]], dim=2)
            rem_k = k_glob_raw[:, :, need_sink:]
            rem_v = v_glob_raw[:, :, need_sink:]

            old_p_len = pk.shape[2]
            stream_k = torch.cat([pk, rem_k], dim=2) if old_p_len > 0 else rem_k
            stream_v = torch.cat([pv, rem_v], dim=2) if old_p_len > 0 else rem_v
            new_n = torch.ones((B, self.global_kv_heads, rem_k.shape[2]), device=dev, dtype=torch.float32)
            stream_n = torch.cat([pn, new_n], dim=2) if old_p_len > 0 else new_n

            new_sum_k, new_sum_v, new_sum_n, sum_rep, (pk, pv, pn), p_rep = self._compress_stream(
                stream_k, stream_v, stream_n, compress_tail=False, mode=active_mode
            )
            if new_sum_k.shape[2] > 0:
                keep = new_sum_n.sum(dim=(0, 1)) > 0
                if not bool(keep.all()):
                    new_sum_k = new_sum_k[:, :, keep]
                    new_sum_v = new_sum_v[:, :, keep]
                    new_sum_n = new_sum_n[:, :, keep]
                    sum_rep = sum_rep[:, keep]

            sink_total = g_sink_k.shape[2]
            n_sink = torch.ones((B, self.global_kv_heads, sink_total), device=dev, dtype=torch.float32)
            parts_k, parts_v, parts_n = [g_sink_k], [g_sink_v], [n_sink]
            if g_sum_k is not None and g_sum_k.shape[2] > 0:
                parts_k.append(g_sum_k)
                parts_v.append(g_sum_v)
                parts_n.append(g_n)
            if new_sum_k.shape[2] > 0:
                parts_k.append(new_sum_k)
                parts_v.append(new_sum_v)
                parts_n.append(new_sum_n)
            if pk.shape[2] > 0:
                parts_k.append(pk)
                parts_v.append(pv)
                parts_n.append(pn)

            k_attn_all = torch.cat(parts_k, dim=2)
            v_attn_all = torch.cat(parts_v, dim=2)
            n_attn_all = torch.cat(parts_n, dim=2)

            if T == 1:
                c_mask = None
            else:
                rep_parts = [
                    (torch.arange(sink_total, device=dev, dtype=torch.long) - sink_l).unsqueeze(0).expand(B, sink_total)
                ]
                if g_sum_k is not None and g_sum_k.shape[2] > 0:
                    rep_parts.append(torch.full((B, g_sum_k.shape[2]), -1, device=dev, dtype=torch.long))
                if new_sum_k.shape[2] > 0:
                    rep_parts.append(sum_rep - old_p_len + need_sink)
                if pk.shape[2] > 0:
                    rep_parts.append(p_rep - old_p_len + need_sink)
                all_rep = torch.cat(rep_parts, dim=1)  # [B, S_all]
                q_pos = torch.arange(T, device=dev, dtype=torch.long).view(1, 1, T, 1)
                c_mask = q_pos >= all_rep.view(B, 1, 1, -1)

            out_glob = self._global_sdpa(q_glob_t, k_attn_all, v_attn_all, n_attn_all, rep_glob, causal_mask=c_mask)

            if new_sum_k.shape[2] > 0:
                g_sum_k = torch.cat([g_sum_k, new_sum_k], dim=2) if g_sum_k is not None and g_sum_k.shape[2] > 0 else new_sum_k
                g_sum_v = torch.cat([g_sum_v, new_sum_v], dim=2) if g_sum_v is not None and g_sum_v.shape[2] > 0 else new_sum_v
                g_n = torch.cat([g_n, new_sum_n], dim=2) if g_n is not None and g_n.shape[2] > 0 else new_sum_n

            if g_sum_k is not None and g_sum_k.shape[2] > self.max_global_cache:
                ka = torch.cat([g_sink_k, g_sum_k], dim=2)
                va = torch.cat([g_sink_v, g_sum_v], dim=2)
                na = torch.cat([n_sink, g_n], dim=2)
                ka, va, na = self._enforce_budget(ka, va, na, sink_total, policy_override=active_mode)
                g_sink_k, g_sink_v = ka[:, :, :sink_total], va[:, :, :sink_total]
                g_sum_k, g_sum_v, g_n = ka[:, :, sink_total:], va[:, :, sink_total:], na[:, :, sink_total:]

            new_glob_cache = (g_sink_k, g_sink_v, g_sum_k, g_sum_v, g_n, (pk, pv, pn))

        out_heads = torch.cat([out_loc, out_glob], dim=1)
        out = self.out_proj(out_heads.transpose(1, 2).contiguous().view(B, T, self.dim))
        new_cache = (new_loc_cache, new_glob_cache) if use_cache else None
        return out, new_cache


class TransformerBlock(nn.Module):
    def __init__(self, cfg: AnchorMassConfig):
        super().__init__()
        self.attn_norm = nn.RMSNorm(cfg.dim)
        self.attn = AnchorMassAttention(cfg)
        self.ffn_norm = nn.RMSNorm(cfg.dim)
        self.ffn_up = nn.Linear(cfg.dim, cfg.ff_dim * 2, bias=False)
        self.ffn_down = nn.Linear(cfg.ff_dim, cfg.dim, bias=False)

    def forward(self, x: torch.Tensor, kv_cache=None, offset: int = 0, use_cache: bool = False):
        attn_out, new_cache = self.attn(self.attn_norm(x), kv_cache=kv_cache, offset=offset, use_cache=use_cache)
        x = x + attn_out
        gate, up = self.ffn_up(self.ffn_norm(x)).chunk(2, dim=-1)
        x = x + self.ffn_down(F.silu(gate) * up)
        return x, new_cache


class AnchorMassForCausalLM(nn.Module):
    def __init__(self, cfg: AnchorMassConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.layers = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.num_layers)])
        self.final_norm = nn.RMSNorm(cfg.dim)
        self.output_proj = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def set_eviction_policy(self, policy: str):
        self.cfg.eviction_policy = policy
        for layer in self.layers:
            layer.attn.eviction_policy = policy

    def set_session_length(self, total_seq_len: int, device: Optional[torch.device] = None):
        for layer in self.layers:
            layer.attn.rope.set_session_length(total_seq_len, device=device)

    def reset_session(self):
        for layer in self.layers:
            layer.attn.rope.reset_session()

    def forward(self, x: torch.Tensor, train_chunk_size: Optional[int] = None) -> torch.Tensor:
        _, T = x.shape
        t_chunk = self.cfg.train_chunk_size if train_chunk_size is None else train_chunk_size

        if T > self.cfg.original_seq_len:
            self.set_session_length(T, device=x.device)

        if t_chunk <= 0 or T <= t_chunk:
            h = self.embed(x)
            for layer in self.layers:
                h, _ = layer(h, use_cache=False)
            return self.output_proj(self.final_norm(h))

        C = getattr(self.cfg, "chunk_size", 64)
        sink = getattr(self.cfg, "sink_tokens", 16)
        first_len = sink + max(1, (t_chunk - sink) // C) * C
        step_len = max(1, t_chunk // C) * C

        caches = [None] * len(self.layers)
        logits_list = []
        pos = 0
        while pos < T:
            end = min(pos + (first_len if pos == 0 else step_len), T)
            h = self.embed(x[:, pos:end])
            for i, layer in enumerate(self.layers):
                h, caches[i] = layer(h, kv_cache=caches[i], offset=pos, use_cache=True)
            logits_list.append(self.output_proj(self.final_norm(h)))
            pos = end

        return torch.cat(logits_list, dim=1)

    @torch.no_grad()
    def prefill_chunked(
        self,
        prompt_ids: torch.Tensor,
        chunk_size: int = 4096,
        total_session_len: Optional[int] = None,
    ):
        self.eval()
        _, P = prompt_ids.shape
        self.set_session_length(total_session_len or P, device=prompt_ids.device)

        C = getattr(self.cfg, "chunk_size", 64)
        sink = getattr(self.cfg, "sink_tokens", 16)
        first_len = sink + max(1, (chunk_size - sink) // C) * C
        step_len = max(1, chunk_size // C) * C

        caches = [None] * len(self.layers)
        h_last = None
        pos = 0
        while pos < P:
            end = min(pos + (first_len if pos == 0 else step_len), P)
            h = self.embed(prompt_ids[:, pos:end])
            for i, layer in enumerate(self.layers):
                layer_fn = getattr(layer, "_orig_mod", layer)
                h, caches[i] = layer_fn(h, kv_cache=caches[i], offset=pos, use_cache=True)
            if end == P:
                h_last = h[:, -1:, :]
            pos = end

        logits = self.output_proj(self.final_norm(h_last)[:, -1, :])
        return caches, logits

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 0.6,
        top_k: int = 40,
        top_p: float = 0.9,
        repetition_penalty: float = 1.15,
        ignore_eos: bool = False,
        eos_id: Optional[int] = None,
    ) -> torch.Tensor:
        self.eval()
        B, P = prompt_ids.shape
        curr_ids = prompt_ids
        target_eos = eos_id if eos_id is not None else self.cfg.eos_id
        total_len = P + max_new_tokens

        try:
            self.set_session_length(total_len, device=prompt_ids.device)
            if P > 4096:
                caches, logits = self.prefill_chunked(prompt_ids, chunk_size=4096, total_session_len=total_len)
            else:
                h = self.embed(curr_ids)
                caches = [None] * len(self.layers)
                for i, layer in enumerate(self.layers):
                    layer_fn = getattr(layer, "_orig_mod", layer)
                    h, caches[i] = layer_fn(h, kv_cache=None, offset=0, use_cache=True)
                logits = self.output_proj(self.final_norm(h)[:, -1, :])

            finished = torch.zeros(B, dtype=torch.bool, device=prompt_ids.device)
            cur_pos = P

            for step in range(max_new_tokens):
                if repetition_penalty != 1.0:
                    for b in range(B):
                        used = torch.unique(curr_ids[b, -64:])
                        logits[b, used] = torch.where(
                            logits[b, used] > 0,
                            logits[b, used] / repetition_penalty,
                            logits[b, used] * repetition_penalty,
                        )

                if temperature == 0.0:
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)
                else:
                    l = logits / temperature
                    if top_k > 0:
                        v, _ = torch.topk(l, min(top_k, l.size(-1)))
                        l[l < v[:, [-1]]] = -float("Inf")
                    if 0.0 < top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(l, descending=True)
                        probs = F.softmax(sorted_logits, dim=-1)
                        sorted_logits[(torch.cumsum(probs, dim=-1) - probs) > top_p] = -float("Inf")
                        l.scatter_(1, sorted_indices, sorted_logits)
                    next_token = torch.multinomial(F.softmax(l, dim=-1), num_samples=1)

                if target_eos is not None and not ignore_eos:
                    next_token = torch.where(finished[:, None], torch.full_like(next_token, target_eos), next_token)

                curr_ids = torch.cat([curr_ids, next_token], dim=1)

                if target_eos is not None and not ignore_eos:
                    finished |= next_token.squeeze(1).eq(target_eos)
                    if bool(finished.all()):
                        break

                if step == max_new_tokens - 1:
                    break

                h = self.embed(next_token)
                for i, layer in enumerate(self.layers):
                    layer_fn = getattr(layer, "_orig_mod", layer)
                    h, caches[i] = layer_fn(h, kv_cache=caches[i], offset=cur_pos, use_cache=True)

                logits = self.output_proj(self.final_norm(h)[:, -1, :])
                cur_pos += 1

            return curr_ids
        finally:
            self.reset_session()


def build_model(params: dict) -> AnchorMassForCausalLM:
    return AnchorMassForCausalLM(AnchorMassConfig.from_dict(params))


def count_params(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    breakdown = {name: sum(p.numel() for p in child.parameters()) for name, child in model.named_children()}
    return {"total": total, "total_m": total / 1e6, "trainable": trainable, "breakdown": breakdown}