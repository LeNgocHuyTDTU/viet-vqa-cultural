"""
evaluate_rl_vs_sft.py
=====================
So sánh đầy đủ 3 model: SFT vs DPO vs PPO

Tự động metrics:
  - VQA Accuracy (Exact Match)
  - BLEU Score
  - ROUGE-L
  - METEOR
  - BERTScore (F1)
  - Reward Score (composite)

Human Eval simulation:
  - Dùng template đánh giá 3 tiêu chí theo thang 1–5:
    (1) Correctness, (2) Fluency, (3) Cultural Relevance
  - In ra bảng so sánh để người đánh giá điền điểm thủ công

Phân tích theo question_type:
  - identification / description / cultural / analysis / comparison

Chạy:
  python evaluate_rl_vs_sft.py
  python evaluate_rl_vs_sft.py --max_samples 200
"""

import os
import json
import argparse
import random
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from tabulate import tabulate
import numpy as np

from config import Config
from models_A import VQA_Model_A
from dataset_rl import PPOVQADataset
from reward_model import VQARewardModel
from evaluation import (
    run_full_evaluation,
    calculate_vqa_accuracy,
    calculate_ngram_metrics,
    calculate_bertscore,
)


# ─────────────────────────────────────────────────────────────────────────────
def load_model(config, vocab_size, ckpt_path: str | None) -> VQA_Model_A:
    model = VQA_Model_A(config, vocab_size).to(config.DEVICE)
    if ckpt_path and os.path.exists(ckpt_path):
        model.load_state_dict(
            torch.load(ckpt_path, map_location=config.DEVICE)
        )
        print(f"  ✅ Loaded: {ckpt_path}")
    else:
        print(f"  ⚠️  {ckpt_path} not found — using random init")
    model.eval()
    return model


def generate_predictions(
    model:      VQA_Model_A,
    dataloader: DataLoader,
    tokenizer,
    config,
    bos_id: int,
    eos_id: int,
    beam_size: int = 4,
) -> tuple[list[str], list[str], list[str]]:
    """Returns (preds, ground_truths, question_types)."""
    all_preds, all_gts, all_types = [], [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="  Generating", leave=False):
            imgs   = batch['image'].to(config.DEVICE)
            q_ids  = batch['question_ids'].to(config.DEVICE)
            q_mask = batch['question_mask'].to(config.DEVICE)

            gen_ids = model.generate(
                imgs, q_ids, q_mask,
                bos_token_id=bos_id,
                eos_token_id=eos_id,
                max_len=config.MAX_ANSWER_LENGTH,
                beam_size=beam_size,
            )
            preds = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
            all_preds.extend(preds)
            all_gts.extend(batch['answer_text'])
            all_types.extend(batch['question_type'])

    return all_preds, all_gts, all_types


def metrics_by_type(
    preds: list[str],
    gts:   list[str],
    types: list[str],
) -> dict[str, dict]:
    """Tính metrics riêng theo question_type."""
    from collections import defaultdict
    grouped = defaultdict(lambda: {'preds': [], 'gts': []})

    for p, g, t in zip(preds, gts, types):
        grouped[t]['preds'].append(p)
        grouped[t]['gts'].append(g)

    result = {}
    for qtype, data in grouped.items():
        p_list = data['preds']
        g_list = data['gts']
        acc    = calculate_vqa_accuracy(p_list, g_list)
        ngram  = calculate_ngram_metrics(p_list, g_list)
        result[qtype] = {
            'n':      len(p_list),
            'acc':    acc,
            'bleu':   ngram['bleu'],
            'rougeL': ngram['rougeL'],
            'meteor': ngram['meteor'],
        }
    return result


def print_comparison_table(results: dict[str, dict]) -> None:
    """In bảng so sánh 3 model."""
    headers = ["Metric", "SFT (baseline)", "DPO", "PPO"]
    rows = [
        ["VQA Accuracy (%)",
         f"{results['sft']['vqa_acc']:.2f}",
         f"{results['dpo']['vqa_acc']:.2f}",
         f"{results['ppo']['vqa_acc']:.2f}"],
        ["BLEU",
         f"{results['sft']['bleu']:.4f}",
         f"{results['dpo']['bleu']:.4f}",
         f"{results['ppo']['bleu']:.4f}"],
        ["ROUGE-L",
         f"{results['sft']['rougeL']:.4f}",
         f"{results['dpo']['rougeL']:.4f}",
         f"{results['ppo']['rougeL']:.4f}"],
        ["METEOR",
         f"{results['sft']['meteor']:.4f}",
         f"{results['dpo']['meteor']:.4f}",
         f"{results['ppo']['meteor']:.4f}"],
        ["BERTScore (F1)",
         f"{results['sft']['bertscore']:.4f}",
         f"{results['dpo']['bertscore']:.4f}",
         f"{results['ppo']['bertscore']:.4f}"],
        ["Reward Score",
         f"{results['sft']['reward']:.4f}",
         f"{results['dpo']['reward']:.4f}",
         f"{results['ppo']['reward']:.4f}"],
    ]

    # Đánh dấu best value (bold workaround)
    for row in rows:
        vals = [float(v) for v in row[1:]]
        best_idx = vals.index(max(vals)) + 1
        row[best_idx] = f"★ {row[best_idx]}"

    print("\n" + "="*70)
    print("  📊 BẢNG SO SÁNH: SFT vs DPO vs PPO  (Automatic Metrics)")
    print("="*70)
    print(tabulate(rows, headers=headers, tablefmt="fancy_grid"))


def print_type_breakdown(
    type_results: dict[str, dict[str, dict]]
) -> None:
    """In breakdown theo question_type cho từng model."""
    all_types = sorted(set(
        t for model_results in type_results.values()
        for t in model_results.keys()
    ))

    print("\n" + "="*70)
    print("  📊 VQA ACCURACY THEO QUESTION TYPE (%)")
    print("="*70)
    headers = ["Question Type", "N", "SFT", "DPO", "PPO"]
    rows = []
    for qt in all_types:
        n   = type_results['sft'].get(qt, {}).get('n', 0)
        sft = type_results['sft'].get(qt, {}).get('acc', 0.0)
        dpo = type_results['dpo'].get(qt, {}).get('acc', 0.0)
        ppo = type_results['ppo'].get(qt, {}).get('acc', 0.0)
        best = max(sft, dpo, ppo)
        rows.append([
            qt, n,
            f"{'★ ' if sft==best else ''}{sft:.2f}",
            f"{'★ ' if dpo==best else ''}{dpo:.2f}",
            f"{'★ ' if ppo==best else ''}{ppo:.2f}",
        ])
    print(tabulate(rows, headers=headers, tablefmt="fancy_grid"))


def generate_human_eval_sheet(
    sft_preds: list[str],
    dpo_preds: list[str],
    ppo_preds: list[str],
    gts:       list[str],
    questions: list[str],
    types:     list[str],
    n_samples: int = 30,
    out_path:  str = "human_eval_sheet.json",
) -> None:
    """
    Tạo file JSON để người đánh giá thực hiện human eval.

    Mỗi entry có:
      - image_path, question, ground_truth
      - 3 response (SFT / DPO / PPO) — thứ tự ngẫu nhiên (blinded)
      - Cột điểm trống: correctness (1-5), fluency (1-5), cultural_relevance (1-5)
    """
    random.seed(0)
    indices = random.sample(range(len(sft_preds)), min(n_samples, len(sft_preds)))

    sheet = []
    for i, idx in enumerate(indices):
        responses = [
            ("SFT",  sft_preds[idx]),
            ("DPO",  dpo_preds[idx]),
            ("PPO",  ppo_preds[idx]),
        ]
        random.shuffle(responses)  # Blind ordering

        entry = {
            "sample_id":    i + 1,
            "question":     questions[idx] if idx < len(questions) else "",
            "ground_truth": gts[idx],
            "question_type": types[idx] if idx < len(types) else "",
            "responses": [
                {
                    "label":               r[0],
                    "response":            r[1],
                    "correctness_1_5":     None,   # người đánh giá điền
                    "fluency_1_5":         None,
                    "cultural_rel_1_5":    None,
                    "notes":               "",
                }
                for r in responses
            ],
            "overall_preference": None,  # SFT / DPO / PPO
        }
        sheet.append(entry)

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(sheet, f, ensure_ascii=False, indent=2)

    print(f"\n📝 Human eval sheet saved: {out_path} ({len(sheet)} samples)")
    print("   Hướng dẫn:")
    print("   1. Mở human_eval_sheet.json")
    print("   2. Điền điểm correctness/fluency/cultural_rel (1–5) cho từng response")
    print("   3. Điền overall_preference = 'SFT' | 'DPO' | 'PPO'")
    print("   4. Chạy analyze_human_eval.py để tổng hợp kết quả\n")


def print_human_eval_summary(sheet_path: str) -> None:
    """Đọc sheet đã điền và tính trung bình."""
    if not os.path.exists(sheet_path):
        return

    with open(sheet_path, 'r', encoding='utf-8') as f:
        sheet = json.load(f)

    # Kiểm tra xem đã điền chưa
    filled = [
        s for s in sheet
        if all(r['correctness_1_5'] is not None for r in s['responses'])
    ]
    if not filled:
        print(f"⚠️  {sheet_path} chưa được điền điểm human eval.")
        return

    scores = {'SFT': [], 'DPO': [], 'PPO': []}
    prefs  = {'SFT': 0, 'DPO': 0, 'PPO': 0}

    for entry in filled:
        for r in entry['responses']:
            label = r['label']
            if r['correctness_1_5'] is not None:
                avg = np.mean([
                    r['correctness_1_5'] or 0,
                    r['fluency_1_5'] or 0,
                    r['cultural_rel_1_5'] or 0,
                ])
                scores[label].append(avg)

        pref = entry.get('overall_preference')
        if pref in prefs:
            prefs[pref] += 1

    print("\n" + "="*70)
    print("  👤 HUMAN EVALUATION RESULTS")
    print("="*70)
    rows = []
    for model in ['SFT', 'DPO', 'PPO']:
        s = scores[model]
        rows.append([
            model,
            len(s),
            f"{np.mean(s):.2f}" if s else "N/A",
            f"{np.std(s):.2f}" if s else "N/A",
            prefs[model],
        ])
    print(tabulate(rows,
                   headers=["Model", "N", "Avg Score (1-5)", "Std", "Preferred Count"],
                   tablefmt="fancy_grid"))


# ─────────────────────────────────────────────────────────────────────────────
def main(max_samples: int = 500):
    config = Config()

    # ── Dataset ─────────────────────────────────────────────────────────────
    full_ds = PPOVQADataset("preference_data.json", config)
    tokenizer  = full_ds.tokenizer
    vocab_size = tokenizer.vocab_size
    pad_id     = tokenizer.pad_token_id or 1
    bos_id     = tokenizer.cls_token_id or 0
    eos_id     = tokenizer.sep_token_id or 2

    # Giới hạn số samples để đánh giá nhanh
    if max_samples and max_samples < len(full_ds):
        indices = random.sample(range(len(full_ds)), max_samples)
        eval_ds = Subset(full_ds, indices)
    else:
        eval_ds = full_ds

    loader = DataLoader(eval_ds, batch_size=8, shuffle=False,
                        num_workers=2, pin_memory=True)

    # ── Load 3 Models ────────────────────────────────────────────────────────
    CHECKPOINTS = {
        'sft': os.path.join(config.CHECKPOINT_DIR, "best_model_A_transformer.pt"),
        'dpo': os.path.join(config.CHECKPOINT_DIR, "best_model_dpo.pt"),
        'ppo': os.path.join(config.CHECKPOINT_DIR, "best_model_ppo.pt"),
    }

    models = {}
    for name, ckpt in CHECKPOINTS.items():
        print(f"\nLoading {name.upper()} model...")
        models[name] = load_model(config, vocab_size, ckpt)

    # ── Generate Predictions ──────────────────────────────────────────────
    reward_model = VQARewardModel(alpha=0.4, beta=0.3, gamma=0.3)
    all_results  = {}
    all_preds    = {}
    shared_gts   = None
    shared_types = None
    shared_qs    = None

    for name, model in models.items():
        print(f"\n{'─'*50}")
        print(f"Evaluating {name.upper()}...")
        preds, gts, types = generate_predictions(
            model, loader, tokenizer, config, bos_id, eos_id,
            beam_size=4,
        )
        all_preds[name] = preds

        if shared_gts is None:
            shared_gts   = gts
            shared_types = types
            # questions 를 dataset에서 꺼내기
            shared_qs = [full_ds.samples[i]['question']
                         for i in (indices if max_samples < len(full_ds) else range(len(full_ds)))]

        # Auto metrics
        metrics = run_full_evaluation(preds, gts, model_label=name.upper())
        rewards = reward_model.compute_batch(preds, gts)
        metrics['reward'] = float(np.mean(rewards))
        all_results[name] = metrics

    # ── Comparison Tables ─────────────────────────────────────────────────
    print_comparison_table(all_results)

    # Type breakdown
    type_results = {
        name: metrics_by_type(all_preds[name], shared_gts, shared_types)
        for name in models
    }
    print_type_breakdown(type_results)

    # ── Human Eval Sheet ─────────────────────────────────────────────────
    generate_human_eval_sheet(
        sft_preds=all_preds['sft'],
        dpo_preds=all_preds['dpo'],
        ppo_preds=all_preds['ppo'],
        gts=shared_gts,
        questions=shared_qs or [""] * len(shared_gts),
        types=shared_types,
        n_samples=30,
        out_path="human_eval_sheet.json",
    )

    # Kiểm tra nếu đã có file điền sẵn
    print_human_eval_summary("human_eval_sheet.json")

    # ── Save full results ─────────────────────────────────────────────────
    final = {
        'auto_metrics':  all_results,
        'type_breakdown': {k: {qt: dict(v) for qt, v in td.items()}
                           for k, td in type_results.items()},
    }
    with open("rl_vs_sft_results.json", "w", encoding="utf-8") as f:
        json.dump(final, f, ensure_ascii=False, indent=2)
    print("\n✅ Full results saved: rl_vs_sft_results.json")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_samples", type=int, default=500)
    args = parser.parse_args()
    main(args.max_samples)
