import json
import random
from pathlib import Path
import numpy as np
import sentencepiece as spm

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_BIN = ROOT_DIR / "data/corpus/token/part_04.bin"
SP_MODEL = ROOT_DIR / "tokenizer/tokenizer.model"

SEED = 20261016

TEMPLATES = {
    "original_prefix": (
        "\nユーザー: 重要機密事項です。今回の秘密パスキーは【{k}】です。忘れないでください。\n"
        "アシスタント: 了解いたしました。秘密パスキー【{k}】を記憶しました。\n",
        "ユーザー: 先ほど記録した秘密パスキーは何ですか？余計な解説は一切含めず、4桁の半角数字のみで答えてください。\nアシスタント: ",
    ),
    "original_no_prefix": (
        "\nユーザー: 重要機密事項です。今回の秘密パスキーは【{k}】です。忘れないでください。\n"
        "アシスタント: 了解いたしました。秘密パスキー【{k}】を記憶しました。\n",
        "ユーザー: 先ほど記録した秘密パスキーは何ですか？パスキーの数字のみ答えてください。\nアシスタント: ",
    ),
    "lexical_paraphrase": (
        "\nユーザー: 忘れないでください。秘密の暗証番号は【{k}】です。\n"
        "アシスタント: わかりました。暗証番号【{k}】を覚えておきます。\n",
        "ユーザー: 先ほどの暗証番号を教えてください。数字だけで答えてください。\nアシスタント: ",
    ),
    "memo_format": (
        "\nメモ: 私の秘密コードは「{k}」です。\n",
        "質問: メモに書かれていた秘密コードは何ですか？数字だけ答えてください。\n回答: ",
    ),
    "dialogue_format": (
        "\n田中: 会議室の解錠コードは【{k}】にしておいたよ。\n佐藤: 了解です！\n",
        "田中: 会議室の解錠コードは何番？数字だけ教えて。\n佐藤: ",
    ),
}

def make_answer(template: str, key: str, style: str) -> str:
    if style == "digits":
        return key
    if template == "lexical_paraphrase":
        return f"暗証番号は【{key}】です。"
    if template == "memo_format":
        return f"秘密コードは「{key}」です。"
    if template == "dialogue_format":
        return f"解錠コードは【{key}】だよ。"
    return f"秘密パスキーは【{key}】です。"


def generate_dataset(target_len: int, n_samples: int, rng: random.Random, raw: np.memmap, sp) -> list:
    samples = []
    names = list(TEMPLATES.keys())
    for i in range(n_samples):
        template = names[i % len(names)]
        style = "digits" if (i % 2 == 0) else "sentence"
        key = str(rng.randint(1000, 9999))
        depth = rng.uniform(0.15, 0.85)

        needle_text = TEMPLATES[template][0].format(k=key)
        query_text = TEMPLATES[template][1]
        answer_text = make_answer(template, key, style)

        needle_ids = sp.encode(needle_text, out_type=int)
        query_ids = sp.encode(query_text, out_type=int)
        answer_ids = sp.encode(answer_text, out_type=int)

        hay_len = target_len - len(needle_ids) - len(query_ids) - len(answer_ids) - 1
        if hay_len <= 0:
            raise ValueError(f"target_len {target_len} too short for {template}")
        start = rng.randint(0, max(1, len(raw) - hay_len - 1))
        hay = raw[start:start + hay_len].astype(int).tolist()
        ins = int(len(hay) * depth)
        prompt_ids = hay[:ins] + needle_ids + hay[ins:] + query_ids
        assert len(prompt_ids) + len(answer_ids) + 1 <= target_len + 8, (len(prompt_ids), target_len)
        prompt_text = sp.decode(prompt_ids)

        samples.append({
            "instruction": prompt_text,
            "input": "",
            "output": answer_text,
            "_meta": {"template": template, "answer_style": style,
                      "target_len": target_len, "depth": round(depth, 3),
                      "bg": "part_04.bin", "bg_off": start, "passkey": key},
        })
    return samples


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_8k", default="data/sft_150M_8k_v2.jsonl")
    ap.add_argument("--out_16k", default="data/sft_150M_16k_v2.jsonl")
    ap.add_argument("--out_32k", default="data/sft_150M_32k_v2.jsonl")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    sp = spm.SentencePieceProcessor(model_file=str(SP_MODEL))
    raw = np.memmap(str(DATA_BIN), dtype=np.uint16, mode="r")
    print(f"SFT bg: {DATA_BIN.name} ({len(raw):,} tok) | eval bg と分離: val.bin")

    plans = [
        (args.out_8k, [2048, 4096, 6144, 8192], 75),
        (args.out_16k, [2048, 4096, 8192, 12288, 16384], 40),
        (args.out_32k, [4096, 8192, 16384, 24576, 32768], 40),
    ]
    for out_rel, lengths, per_len in plans:
        all_s = []
        for L in lengths:
            all_s += generate_dataset(L, per_len, rng, raw, sp)
        rng.shuffle(all_s)
        from collections import Counter
        print(f"-- {out_rel}: {len(all_s)}件 "
              f"{dict(Counter(s['_meta']['template'] for s in all_s))} "
              f"{dict(Counter(s['_meta']['answer_style'] for s in all_s))}")
        out = ROOT_DIR / out_rel
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            for s in all_s:
                f.write(json.dumps({k: s[k] for k in ("instruction", "input", "output")}, ensure_ascii=False) + "\n")
        with open(str(out) + ".meta.jsonl", "w", encoding="utf-8") as f:
            for s in all_s:
                f.write(json.dumps(s["_meta"], ensure_ascii=False) + "\n")
        print(f"   saved {out} + .meta.jsonl")

    print("Done. bg=part_04.bin (eval val.bin と非重複)")


if __name__ == "__main__":
    main()
