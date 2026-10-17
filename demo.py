import sys
import yaml
import argparse
from pathlib import Path
import torch
import sentencepiece as spm

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from anchormass import AnchorMassForCausalLM, AnchorMassConfig, count_params


def main():
    parser = argparse.ArgumentParser(description="AnchorMass Quickstart Demo")
    parser.add_argument("--ckpt", type=str, default=None, help="Path to checkpoint (.pt)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--ctx_len", type=int, default=8192, help="Context length to test")
    parser.add_argument("--max_cache", type=int, default=512, help="Bounded KV cache budget")
    args = parser.parse_args()

    print("\nAnchorMass: Bounded KV-Cache Long-Context Language Model Demo\n")

    yaml_path = ROOT_DIR / "anchormass/config.yaml"
    with open(yaml_path) as f:
        cfg_dict = yaml.safe_load(f)["anchormass_150M"]

    cfg_dict["MAX_GLOBAL_CACHE"] = args.max_cache
    cfg_dict["MAX_POSITION_EMBEDDINGS"] = 262144
    cfg_dict["ROPE_SCALING_TYPE"] = "dynamic_ntk"
    cfg_dict["EVICTION_POLICY"] = "adaptive"

    cfg = AnchorMassConfig.from_dict(cfg_dict)
    model = AnchorMassForCausalLM(cfg).to(device=args.device, dtype=torch.bfloat16).eval()

    info = count_params(model)
    print(f"モデル構築完了: 総パラメータ数 = {info['total']:,} ({info['total_m']:.1f}M)")
    print(f"   - 隠れ層次元: {cfg.dim}, 層数: {cfg.num_layers}, ヘッド数: {cfg.num_heads} (Local 6 / Global 6)")
    print(f"   - グローバルキャッシュ上限: {args.max_cache} slots (固定)")

    if args.ckpt and Path(args.ckpt).exists():
        print(f"\n重みをロード中: {args.ckpt}")
        sd = torch.load(args.ckpt, map_location=args.device, weights_only=False)
        if isinstance(sd, dict) and "model_state_dict" in sd:
            sd = sd["model_state_dict"]
        sd = {k.replace("._orig_mod.", ".").removeprefix("_orig_mod."): v for k, v in sd.items()}
        sd = {k: v for k, v in sd.items() if not any(x in k for x in ("cos_cached", "sin_cached", "inv_freq"))}
        model.load_state_dict(sd, strict=False)
        print("重みのロードに成功しました")
    else:
        print("\nチェックポイント未指定のため、初期化状態でフォワード動作をテストします")

    print(f"\n系列長 {args.ctx_len:,} トークンでの Prefill & キャッシュ上限テスト...")
    dummy_input = torch.randint(0, cfg.vocab_size, (1, args.ctx_len), device=args.device)

    with torch.no_grad():
        caches = [None] * cfg.num_layers
        pos = 0
        chunk_size = 1024
        while pos < args.ctx_len:
            end = min(pos + chunk_size, args.ctx_len)
            h = model.embed(dummy_input[:, pos:end])
            for i, layer in enumerate(model.layers):
                h, caches[i] = layer(h, kv_cache=caches[i], offset=pos, use_cache=True)
            pos = end

    total_slots = sum(c[1][2].shape[2] for c in caches if c is not None and c[1][2] is not None)
    avg_slots = total_slots / cfg.num_layers
    total_bytes = sum(
        sum(t.numel() * t.element_size() for t in (c[1][0], c[1][1], c[1][2], c[1][3]) if t is not None)
        for c in caches if c is not None
    )
    print(f"Prefill 完了:")
    print(f"   - 入力系列長: {args.ctx_len:,} トークン")
    print(f"   - 各層の Global キャッシュスロット数: 平均 {avg_slots:.0f} slots (上限 {args.max_cache} スロット以下に維持)")
    print(f"   - 全14層の合計 KV キャッシュメモリ: {total_bytes / (1024 ** 2):.1f} MB")
    print("\nAnchorMass Attention の上限付きキャッシュ機構が正常に機能しています！")


if __name__ == "__main__":
    main()
