import os
import sys
import json
import random
import time
import re
import yaml
import argparse
from pathlib import Path

import numpy as np
import torch
import sentencepiece as spm
from transformers import AutoTokenizer, AutoModelForCausalLM

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from anchormass.standard_model import GPTLanguageModel, GPTConfig
from anchormass.model import AnchorMassForCausalLM, AnchorMassConfig
from eval.utils import expand_standard_rope

BENCHMARK_QUESTIONS = [
    {"id": 1, "cat": "knowledge", "q": "日本の首都はどこですか？", "ref": "日本の首都は東京です。"},
    {"id": 2, "cat": "knowledge", "q": "織田信長を本能寺で討った戦国武将は誰ですか？", "ref": "明智光秀です。"},
    {"id": 3, "cat": "knowledge", "q": "富士山の標高はおよそ何メートルですか？", "ref": "およそ3,776メートルです。"},
    {"id": 4, "cat": "knowledge", "q": "太陽系で最も太陽に近い惑星は何ですか？", "ref": "水星（マーキュリー）です。"},
    {"id": 5, "cat": "knowledge", "q": "光合成で植物が吸収する気体は何ですか？", "ref": "二酸化炭素（CO2）です。"},
    {"id": 6, "cat": "knowledge", "q": "江戸幕府を開いた初代将軍は誰ですか？", "ref": "徳川家康です。"},
    {"id": 7, "cat": "knowledge", "q": "水の化学式を教えてください。", "ref": "H2O です。"},
    {"id": 8, "cat": "knowledge", "q": "日本で最も長い川は何ですか？", "ref": "信濃川です。"},
    {"id": 9, "cat": "knowledge", "q": "人工知能における「ディープラーニング」とは何ですか？簡潔に答えてください。", "ref": "多層のニューラルネットワークを用いてデータの特徴を自動学習する機械学習手法です。"},
    {"id": 10, "cat": "knowledge", "q": "コンピュータのCPUとは、何の略ですか？", "ref": "Central Processing Unit（中央演算処理装置）の略です。"},
    {"id": 11, "cat": "knowledge", "q": "夏目漱石の代表作「吾輩は猫である」の冒頭はどのように始まりますか？", "ref": "「吾輩は猫である。名前はまだ無い。」です。"},
    {"id": 12, "cat": "knowledge", "q": "元素記号「Au」は何を表していますか？", "ref": "金（Gold）を表しています。"},
    {"id": 13, "cat": "knowledge", "q": "世界で最も面積の広い国はどこですか？", "ref": "ロシアです。"},
    {"id": 14, "cat": "knowledge", "q": "人間の成人の骨はおよそいくつありますか？", "ref": "およそ206個です。"},
    {"id": 15, "cat": "knowledge", "q": "1日は何秒ですか？計算して答えてください。", "ref": "86,400秒（24時間 × 60分 × 60秒）です。"},
    {"id": 16, "cat": "knowledge", "q": "名画「モナ・リザ」を描いた画家は誰ですか？", "ref": "レオナルド・ダ・ヴィンチです。"},
    {"id": 17, "cat": "knowledge", "q": "野球において、1チームがフィールドに出る人数は何人ですか？", "ref": "9人です。"},
    {"id": 18, "cat": "knowledge", "q": "徳川家康が江戸幕府を開いたのは西暦何年ですか？", "ref": "西暦1603年です。"},
    {"id": 19, "cat": "knowledge", "q": "円周率を小数点以下2桁まで教えてください。", "ref": "3.14 です。"},
    {"id": 20, "cat": "knowledge", "q": "パソコンのキーボードで、入力や実行を確定するためによく使うキーは何ですか？", "ref": "Enterキー（Returnキー）です。"},
    {"id": 21, "cat": "knowledge", "q": "地球の自転周期はおよそ何時間ですか？", "ref": "およそ24時間（約23時間56分4秒）です。"},
    {"id": 22, "cat": "knowledge", "q": "カエルは両生類ですか、それとも爬虫類ですか？", "ref": "両生類です。"},
    {"id": 23, "cat": "knowledge", "q": "英語のアルファベットは全部で何文字ありますか？", "ref": "26文字です。"},
    {"id": 24, "cat": "knowledge", "q": "紫式部が書いたとされる平安時代の代表的な物語は何ですか？", "ref": "源氏物語です。"},
    {"id": 25, "cat": "knowledge", "q": "太陽系の惑星の中で一番大きいのは何ですか？", "ref": "木星です。"},
    {"id": 26, "cat": "reasoning", "q": "おいしいオムライスを作るコツを教えてください。", "ref": "強火で手早く卵を半熟に仕上げることや、ケチャップライスの水分をしっかり飛ばすことです。"},
    {"id": 27, "cat": "reasoning", "q": "こんにちは！自己紹介してください。", "ref": "自然な挨拶とAIアシスタントとしての役割を述べる。"},
    {"id": 28, "cat": "reasoning", "q": "新社会人に向けて、仕事で大切にすべきアドバイスを2つ教えてください。", "ref": "報告・連絡・相談（ホウレンソウ）の徹底と、健康・体調管理など2つの具体的助言。"},
    {"id": 29, "cat": "reasoning", "q": "犬と猫をペットとして飼うときの主な違いは何ですか？", "ref": "散歩の必要性やしつけのしやすさ、独立性の違いなど。"},
    {"id": 30, "cat": "reasoning", "q": "「失敗は成功のもと」ということわざの意味を分かりやすく説明してください。", "ref": "失敗の原因を反省して改善すれば、次の成功につながるという意味。"},
    {"id": 31, "cat": "reasoning", "q": "旅行の計画を立てる際、最初に決めるべきことは何ですか？理由も教えてください。", "ref": "日程や予算、目的地などの優先順位とその理由。"},
    {"id": 32, "cat": "reasoning", "q": "朝すっきりと目覚めるための良い習慣を教えてください。", "ref": "太陽の光を浴びる、コップ一杯の水を飲む、決まった時間に起きるなど。"},
    {"id": 33, "cat": "reasoning", "q": "カレーライスの基本的な作り方をステップ順に教えてください。", "ref": "具材を切る→炒める→煮込む→火を止めてルーを溶かす→再び煮込む。"},
    {"id": 34, "cat": "reasoning", "q": "あなたは宇宙飛行士です。宇宙から地球を見たときの感想を述べてください。", "ref": "青く美しい地球の姿や国境のない一体感などの情景描写。"},
    {"id": 35, "cat": "reasoning", "q": "読書をすることのメリットを簡潔に教えてください。", "ref": "知識の獲得、語彙力・集中力の向上、他者の視点の擬似体験など。"},
    {"id": 36, "cat": "reasoning", "q": "地球温暖化を防ぐために私たちが日常生活でできることを3つ挙げてください。", "ref": "節電、マイバッグの利用、公共交通機関の活用など具体的な3点。"},
    {"id": 37, "cat": "reasoning", "q": "なぜ空は青く見えるのですか？小学生にもわかるように簡単に説明してください。", "ref": "太陽の光のうち青い光が空気の粒にぶつかって散らばる（散乱）ため。"},
    {"id": 38, "cat": "reasoning", "q": "健康的な食生活を送るためのポイントを教えてください。", "ref": "主食・主菜・副菜のバランス、野菜の摂取、塩分・糖分の控えめなど。"},
    {"id": 39, "cat": "reasoning", "q": "もしタイムマシンがあったら、過去と未来どちらに行きたいですか？理由も添えてください。", "ref": "過去または未来のどちらかを選び、一貫した論理的な理由を述べる。"},
    {"id": 40, "cat": "reasoning", "q": "長時間のデスクワークによる運動不足を解消するための、簡単なストレッチを教えてください。", "ref": "肩甲骨回しや首・手首のストレッチなど座ったままできる動作。"},
    {"id": 41, "cat": "reasoning", "q": "プログラミングを学ぶことのメリットは何だと考えますか？", "ref": "論理的思考力の養成、業務の自動化・効率化、新しいアイデアを具現化できること。"},
    {"id": 42, "cat": "reasoning", "q": "お金を節約するための効果的で実践的な方法を教えてください。", "ref": "固定費の見直し（通信費やサブスク）、家計簿による可視化、自炊など。"},
    {"id": 43, "cat": "reasoning", "q": "友達と些細なことで喧嘩してしまったとき、どのように仲直りするのが良いでしょうか？", "ref": "冷静になってから素直に謝罪する、相手の気持ちを聞くなど。"},
    {"id": 44, "cat": "reasoning", "q": "全く新しい言語を学ぶための効果的なアプローチは何ですか？", "ref": "基本文法の習得、日常会話の音声シャドーイング、実践的な会話機会の確保など。"},
    {"id": 45, "cat": "reasoning", "q": "なぜ人間は毎日睡眠を取る必要があるのですか？", "ref": "脳と身体の疲労回復、記憶の定着や整理、免疫力の維持のため。"},
    {"id": 46, "cat": "reasoning", "q": "スマートフォンやSNSの使いすぎによるデメリットと、その対策を教えてください。", "ref": "睡眠不足や集中力低下などのデメリットと、スクリーンタイム制限などの対策。"},
    {"id": 47, "cat": "reasoning", "q": "あなたが思う「幸せな人生」とはどのようなものですか？", "ref": "心身の健康や良好な人間関係、自己実現などについての肯定的な考察。"},
    {"id": 48, "cat": "reasoning", "q": "将来の夢や目標が見つからない人へのアドバイスをお願いします。", "ref": "目の前のことに興味を持つ、小さな行動から始める、焦る必要はないという励まし。"},
    {"id": 49, "cat": "reasoning", "q": "映画館で映画を見るのと、家で映画を見るの、どちらが好きですか？それぞれの良さを比較して答えてください。", "ref": "映画館の迫力ある音響・映像と、自宅のリラックス・自由さを対比。"},
    {"id": 50, "cat": "reasoning", "q": "強いストレスを感じたときの、おすすめのリラックス方法を教えてください。", "ref": "深呼吸、軽い散歩、ぬるめのお風呂に入る、好きな音楽を聴くなど。"},
]


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
def generate_text(model, prompt_ids, max_new_tokens=100, is_anchormass=False, eos_ids=(2, 3)):
    device = prompt_ids.device
    curr_ids = prompt_ids.clone()
    gen = []

    for _ in range(max_new_tokens):
        P = curr_ids.shape[1]
        model.set_session_length(P, device=device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            if is_anchormass:
                h = model.embed(curr_ids)
                for layer in model.layers:
                    layer_fn = getattr(layer, "_orig_mod", layer)
                    h, _ = layer_fn(h, kv_cache=None, offset=0, use_cache=False)
                logits = model.output_proj(model.final_norm(h)[:, -1, :])
            else:
                logits = model(curr_ids)[:, -1, :]

        if len(gen) > 0:
            for prev in set(gen[-32:]):
                if logits[0, prev] > 0:
                    logits[0, prev] /= 1.15
                else:
                    logits[0, prev] *= 1.15

        next_tok = torch.argmax(logits, dim=-1).item()
        if next_tok in eos_ids:
            break
        gen.append(next_tok)
        curr_ids = torch.cat([curr_ids, torch.tensor([[next_tok]], device=device)], dim=1)

    return gen


def build_pairwise_judge_prompt(question: str, ans_a: str, ans_b: str, ref: str) -> str:
    ans_a_text = ans_a.strip() if ans_a and ans_a.strip() else "（回答なし・空白）"
    ans_b_text = ans_b.strip() if ans_b and ans_b.strip() else "（回答なし・空白）"

    return f"""あなたは厳格かつ公正なAI採点官です。以下の【質問】に対する【回答A】と【回答B】の品質を【模範解答】をもとにそれぞれ10点満点で採点してください。

【質問】: {question}
【模範解答】: {ref}

【回答A】: {ans_a_text}
【回答B】: {ans_b_text}

【採点基準】
- 模範解答に合致した正確で自然な回答: 8〜10点
- 部分的・やや不十分だが文脈には沿う回答: 4〜7点
- 無回答、空白、明らかな誤答、オウム返し、破綻: 0〜3点

以下のフォーマットに従って必ず出力してください：
回答Aの得点: [0〜10の数字]点
回答Bの得点: [0〜10の数字]点
勝者: [[A]] または [[B]] または [[TIE]]
理由: [1行で簡潔に]
""".strip()


def parse_judgment(text: str) -> str:
    m = re.findall(r"\[\[?\s*(A|B|TIE)\s*\]?\]", text, re.IGNORECASE)
    if m:
        return m[-1].upper()

    m = re.findall(r"(?:勝者|判定|結果|結論|Winner)[:：\s]+(?:回答)?\s*(A|B|TIE|同等|引き分け)", text, re.IGNORECASE)
    if m:
        val = m[-1].upper()
        if val in ("同等", "引き分け"):
            return "TIE"
        return val

    sa = re.search(r"回答\s*A(?:の得点)?[:：\s]*\[?(\d+)\]?", text)
    sb = re.search(r"回答\s*B(?:の得点)?[:：\s]*\[?(\d+)\]?", text)
    if sa and sb:
        score_a = int(sa.group(1))
        score_b = int(sb.group(1))
        if score_a > score_b:
            return "A"
        elif score_b > score_a:
            return "B"
        return "TIE"

    is_a_win = bool(re.search(r"回答\s*A\s*が(?:優れて|良い|正解|正しい|適切|勝って)", text))
    is_b_win = bool(re.search(r"回答\s*B\s*が(?:優れて|良い|正解|正しい|適切|勝って)", text))
    if is_a_win and not is_b_win:
        return "A"
    if is_b_win and not is_a_win:
        return "B"

    tokens = re.findall(r"\b(A|B|TIE)\b", text)
    if tokens:
        return tokens[-1].upper()

    return "TIE"


def main():
    parser = argparse.ArgumentParser(description="Pairwise Blind Evaluation Suite")
    parser.add_argument("--std_ckpt", type=str, default="checkpoints/standard/tf_1_150M_chat_sft.pt")
    parser.add_argument("--am_ckpt", type=str, default="checkpoints/anchormass/lb_11_150M_chat_sft.pt")
    parser.add_argument("--judge_model", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="HuggingFace model for Judge (e.g. 7B, 14B, 32B, 72B)")
    parser.add_argument("--skip_judge", action="store_true", help="Only generate answers and human review sheet, skip automatic judge")
    parser.add_argument("--limit", type=int, default=0, help="Number of questions (0 = all 50)")
    parser.add_argument("--report_out", type=str, default="logs/evaluation_pairwise_blind_summary.md")
    parser.add_argument("--html_out", type=str, default="logs/human_blind_review.html")
    parser.add_argument("--pairs_file", type=str, default="", help="Path to pre-generated blind pairs JSON (skips 150M inference and runs judge directly)")
    parser.add_argument("--save_pairs", type=str, default="logs/blind_pairs.json", help="Path to save generated blind pairs JSON")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.pairs_file and os.path.exists(args.pairs_file):
        print(f"\nLoading pre-generated blind pairs from: {args.pairs_file}")
        with open(args.pairs_file, "r", encoding="utf-8") as f:
            blind_pairs = json.load(f)
        questions = [{"id": p["id"], "q": p["q"], "ref": p["ref"], "cat": p["cat"]} for p in blind_pairs]
        if args.limit > 0:
            blind_pairs = blind_pairs[:args.limit]
            questions = questions[:args.limit]
    else:
        sp_path = ROOT_DIR / "tokenizer/tokenizer.model"
        sp = spm.SentencePieceProcessor(model_file=str(sp_path))
        questions = BENCHMARK_QUESTIONS[:args.limit] if args.limit > 0 else BENCHMARK_QUESTIONS

        print(f"Pairwise Blind Benchmark ({len(questions)} Questions)")
        print(f"Standard  : {args.std_ckpt}")
        print(f"AnchorMass: {args.am_ckpt}")

        sys_prompt = "あなたは親切なAIアシスタントです。指示に対して自然で正確な日本語で回答してください。"

        print("\n[Phase 1/3] Generating Standard Answers...")
        std_m = load_standard(device, str(ROOT_DIR / args.std_ckpt))
        std_ans = {}
        for item in questions:
            p = f"System: {sys_prompt}\n\nユーザー: {item['q']}\n\nアシスタント: "
            ids = torch.tensor([sp.encode(p, out_type=int)], device=device)
            gen = generate_text(std_m, ids, max_new_tokens=100, is_anchormass=False)
            std_ans[item["id"]] = sp.decode(gen).strip()
        del std_m
        torch.cuda.empty_cache()

        print("\n[Phase 2/3] Generating AnchorMass Answers...")
        am_m = load_anchormass(device, str(ROOT_DIR / args.am_ckpt))
        am_ans = {}
        for item in questions:
            p = f"System: {sys_prompt}\n\nユーザー: {item['q']}\n\nアシスタント: "
            ids = torch.tensor([sp.encode(p, out_type=int)], device=device)
            gen = generate_text(am_m, ids, max_new_tokens=100, is_anchormass=True)
            am_ans[item["id"]] = sp.decode(gen).strip()
        del am_m
        torch.cuda.empty_cache()

        blind_pairs = []
        for item in questions:
            qid = item["id"]
            s_txt = std_ans[qid]
            a_txt = am_ans[qid]
            swap = (random.random() < 0.5)
            if swap:
                ans_a, ans_b = a_txt, s_txt
                model_a, model_b = "anchormass", "standard"
            else:
                ans_a, ans_b = s_txt, a_txt
                model_a, model_b = "standard", "anchormass"

            blind_pairs.append({
                "id": qid,
                "cat": item["cat"],
                "q": item["q"],
                "ref": item["ref"],
                "ans_a": ans_a,
                "ans_b": ans_b,
                "model_a": model_a,
                "model_b": model_b,
            })

        if args.save_pairs:
            save_path = ROOT_DIR / args.save_pairs
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(blind_pairs, f, ensure_ascii=False, indent=2)
            print(f"Blind pairs saved to: {save_path}")

    judge_results = []
    am_wins = 0
    std_wins = 0
    ties = 0

    if not args.skip_judge:
        print(f"\n[Phase 3/3] Loading Judge Model ({args.judge_model})...")
        try:
            j_tok = AutoTokenizer.from_pretrained(args.judge_model)
            j_model = AutoModelForCausalLM.from_pretrained(
                args.judge_model, torch_dtype=torch.bfloat16
            ).to(device).eval()

            print("Judging Blind A/B pairs (with Positional Swap)...")
            for pair in blind_pairs:
                qid = pair["id"]
                p1 = build_pairwise_judge_prompt(pair["q"], pair["ans_a"], pair["ans_b"], pair["ref"])
                c1 = j_tok.apply_chat_template([{"role": "user", "content": p1}], tokenize=False, add_generation_prompt=True)
                in1 = j_tok(c1, return_tensors="pt").to(device)
                with torch.no_grad():
                    out1 = j_model.generate(**in1, max_new_tokens=100, temperature=0.0, do_sample=False)
                res1 = j_tok.decode(out1[0][in1.input_ids.shape[1]:], skip_special_tokens=True).strip()
                dec1 = parse_judgment(res1)

                p2 = build_pairwise_judge_prompt(pair["q"], pair["ans_b"], pair["ans_a"], pair["ref"])
                c2 = j_tok.apply_chat_template([{"role": "user", "content": p2}], tokenize=False, add_generation_prompt=True)
                in2 = j_tok(c2, return_tensors="pt").to(device)
                with torch.no_grad():
                    out2 = j_model.generate(**in2, max_new_tokens=100, temperature=0.0, do_sample=False)
                res2 = j_tok.decode(out2[0][in2.input_ids.shape[1]:], skip_special_tokens=True).strip()
                dec2 = parse_judgment(res2)

                if dec1 == "A" and dec2 == "B":
                    winner_model = pair["model_a"]
                elif dec1 == "B" and dec2 == "A":
                    winner_model = pair["model_b"]
                else:
                    winner_model = "TIE"

                if winner_model in ("anchormass", "am"):
                    am_wins += 1
                elif winner_model in ("standard", "std"):
                    std_wins += 1
                else:
                    ties += 1

                judge_results.append({
                    "id": qid,
                    "q": pair["q"],
                    "winner": winner_model,
                    "dec1": dec1,
                    "dec2": dec2,
                    "res1": res1,
                    "res2": res2,
                })
                r1_s = res1.replace('\n', ' ')[:30]
                r2_s = res2.replace('\n', ' ')[:30]
                print(f"  Q{qid:2d}: Winner={winner_model:8s} | 判定1={dec1:3s} ({r1_s}) | 判定2={dec2:3s} ({r2_s})")

            del j_model, j_tok
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"Automatic judge failed: {e}")
            print("Skipping automatic judge and generating human review sheet.")

    md = []
    md.append("# 150M 完全統制対照 Pairwise ブラインド評価成果報告書\n")
    md.append(f"- **実施日時**: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    md.append(f"- **評価問題数**: {len(questions)} 問 (知識 25問, 対話・論理 25問)")
    md.append(f"- **プロトコル**: A/B ランダムブラインド割り当て + Positional Swap (位置バイアス排除)")
    md.append(f"- **Standard ckpt**: `{args.std_ckpt}`")
    md.append(f"- **AnchorMass ckpt**: `{args.am_ckpt}`")
    md.append(f"- **審査モデル**: `{args.judge_model}` (Greedy, temp=0.0, max_new_tokens=100, swap一致のみWin)")
    md.append(f"- **Seed**: {args.seed}")

    if not args.skip_judge and len(judge_results) > 0:
        total = len(judge_results)
        a_rate = am_wins / total * 100
        s_rate = std_wins / total * 100
        t_rate = ties / total * 100

        md.append("## 1. 自動ブラインド審査 (Pairwise 勝敗・同等率)\n")
        md.append("| 判定区分 | 件数 | 割合 |")
        md.append("| :--- | :---: | :---: |")
        md.append(f"| **完全同等 (TIE)** | **{ties}** | **{t_rate:.1f}%** |")
        md.append(f"| **AnchorMass 優位 (Win)** | {am_wins} | {a_rate:.1f}% |")
        md.append(f"| **Standard 優位 (Win)** | {std_wins} | {s_rate:.1f}% |\n")

    rep_path = ROOT_DIR / args.report_out
    rep_path.parent.mkdir(parents=True, exist_ok=True)
    with open(rep_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print(f"Summary report saved to: {rep_path}")


if __name__ == "__main__":
    main()
