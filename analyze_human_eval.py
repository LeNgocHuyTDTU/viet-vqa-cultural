"""
analyze_human_eval.py
=====================
Phân tích kết quả human evaluation sau khi đã điền điểm vào
human_eval_sheet.json.

Xuất:
  - Bảng điểm trung bình theo model (Correctness / Fluency / Cultural)
  - Win/Loss/Tie matrix
  - Kết luận tổng hợp RL vs SFT

Chạy sau khi điền xong human_eval_sheet.json:
  python analyze_human_eval.py
  python analyze_human_eval.py --sheet human_eval_sheet.json
"""

import json
import argparse
import numpy as np
from tabulate import tabulate
from collections import defaultdict


def load_sheet(path: str) -> list[dict]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def analyze(sheet: list[dict]) -> None:
    filled = [
        s for s in sheet
        if all(
            r.get('correctness_1_5') is not None
            for r in s['responses']
        )
    ]

    if not filled:
        print("⚠️  Chưa có mẫu nào được điền đầy đủ điểm.")
        print("    Hãy mở human_eval_sheet.json và điền:")
        print("    correctness_1_5, fluency_1_5, cultural_rel_1_5 cho mỗi response")
        print("    overall_preference = 'SFT' | 'DPO' | 'PPO'")
        return

    print(f"\n✅ Đã phân tích {len(filled)}/{len(sheet)} mẫu đã điền điểm\n")

    # ── Per-model scores ──────────────────────────────────────────────────
    scores = defaultdict(lambda: {'corr': [], 'flu': [], 'cult': [], 'avg': []})
    prefs  = defaultdict(int)
    type_scores = defaultdict(lambda: defaultdict(list))

    for entry in filled:
        qt = entry.get('question_type', 'unknown')
        for r in entry['responses']:
            lbl  = r['label']
            corr = r.get('correctness_1_5') or 0
            flu  = r.get('fluency_1_5') or 0
            cult = r.get('cultural_rel_1_5') or 0
            avg  = np.mean([corr, flu, cult])

            scores[lbl]['corr'].append(corr)
            scores[lbl]['flu'].append(flu)
            scores[lbl]['cult'].append(cult)
            scores[lbl]['avg'].append(avg)
            type_scores[qt][lbl].append(avg)

        pref = entry.get('overall_preference')
        if pref:
            prefs[pref] += 1

    # ── Table 1: Average scores ───────────────────────────────────────────
    headers1 = ["Model", "Correctness", "Fluency", "Cultural Rel.", "Average", "Preferred"]
    rows1 = []
    for model in ['SFT', 'DPO', 'PPO']:
        s = scores[model]
        rows1.append([
            model,
            f"{np.mean(s['corr']):.2f}" if s['corr'] else "–",
            f"{np.mean(s['flu']):.2f}"  if s['flu']  else "–",
            f"{np.mean(s['cult']):.2f}" if s['cult'] else "–",
            f"{np.mean(s['avg']):.2f}"  if s['avg']  else "–",
            prefs.get(model, 0),
        ])

    # Highlight best avg
    avgs = [float(r[4]) if r[4] != "–" else 0 for r in rows1]
    best_avg = max(avgs)
    for i, row in enumerate(rows1):
        if avgs[i] == best_avg:
            row[0] = f"★ {row[0]}"

    print("="*70)
    print("  👤 HUMAN EVALUATION — DETAILED SCORES (1–5)")
    print("="*70)
    print(tabulate(rows1, headers=headers1, tablefmt="fancy_grid"))

    # ── Table 2: Win-Loss matrix ──────────────────────────────────────────
    models = ['SFT', 'DPO', 'PPO']
    model_avgs = {m: scores[m]['avg'] for m in models}

    wl_matrix = defaultdict(lambda: defaultdict(int))
    for entry in filled:
        resp_dict = {r['label']: np.mean([
            r.get('correctness_1_5') or 0,
            r.get('fluency_1_5') or 0,
            r.get('cultural_rel_1_5') or 0,
        ]) for r in entry['responses']}

        for i, m1 in enumerate(models):
            for m2 in models:
                if m1 == m2: continue
                v1 = resp_dict.get(m1, 0)
                v2 = resp_dict.get(m2, 0)
                if v1 > v2:
                    wl_matrix[m1]['win'] += 1
                elif v1 < v2:
                    wl_matrix[m1]['loss'] += 1
                else:
                    wl_matrix[m1]['tie'] += 1

    headers2 = ["Model", "Wins", "Losses", "Ties", "Win Rate"]
    rows2 = []
    for model in models:
        w = wl_matrix[model]['win']
        l = wl_matrix[model]['loss']
        t = wl_matrix[model]['tie']
        total = w + l + t
        wr = f"{w/total*100:.1f}%" if total > 0 else "–"
        rows2.append([model, w, l, t, wr])

    print("\n" + "="*70)
    print("  🏆 WIN / LOSS / TIE MATRIX")
    print("="*70)
    print(tabulate(rows2, headers=headers2, tablefmt="fancy_grid"))

    # ── Table 3: By question type ─────────────────────────────────────────
    all_types = sorted(type_scores.keys())
    if all_types:
        headers3 = ["Question Type"] + models
        rows3 = []
        for qt in all_types:
            row = [qt]
            type_avgs = []
            for model in models:
                vals = type_scores[qt].get(model, [])
                avg  = np.mean(vals) if vals else 0.0
                type_avgs.append(avg)
                row.append(f"{avg:.2f}" if vals else "–")
            best = max(type_avgs)
            for i, avg in enumerate(type_avgs):
                if avg == best and avg > 0:
                    row[i+1] = f"★ {row[i+1]}"
            rows3.append(row)

        print("\n" + "="*70)
        print("  📊 HUMAN SCORES BY QUESTION TYPE (avg 1–5)")
        print("="*70)
        print(tabulate(rows3, headers=headers3, tablefmt="fancy_grid"))

    # ── Conclusion ────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("  📝 TỔNG KẾT SO SÁNH RL vs SFT")
    print("="*70)

    sft_avg = np.mean(scores['SFT']['avg']) if scores['SFT']['avg'] else 0
    dpo_avg = np.mean(scores['DPO']['avg']) if scores['DPO']['avg'] else 0
    ppo_avg = np.mean(scores['PPO']['avg']) if scores['PPO']['avg'] else 0

    for name, avg in [('DPO', dpo_avg), ('PPO', ppo_avg)]:
        delta = avg - sft_avg
        direction = "cải thiện ✅" if delta > 0.05 else \
                    "giảm nhẹ ⚠️" if delta < -0.05 else "tương đương ➡️"
        print(f"  {name} vs SFT: Δ = {delta:+.2f} → {direction}")

    best_model = max([('SFT', sft_avg), ('DPO', dpo_avg), ('PPO', ppo_avg)],
                     key=lambda x: x[1])
    print(f"\n  🏅 Model tốt nhất (human eval): {best_model[0]} "
          f"(avg score = {best_model[1]:.2f}/5.00)")

    most_preferred = max(prefs.items(), key=lambda x: x[1]) if prefs else None
    if most_preferred:
        print(f"  👍 Model được ưu tiên nhất: {most_preferred[0]} "
              f"({most_preferred[1]}/{len(filled)} lần chọn)")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sheet", default="human_eval_sheet.json")
    args = parser.parse_args()

    sheet = load_sheet(args.sheet)
    analyze(sheet)
