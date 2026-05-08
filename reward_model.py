"""
reward_model.py
===============
Reward function tổng hợp dùng cho PPO:

  reward(pred, gt) = α · ExactMatch(pred, gt)
                   + β · ROUGE-L(pred, gt)
                   + γ · BERTScore(pred, gt)    ← semantic
                   - δ · LengthPenalty(pred)    ← tránh câu quá ngắn/dài

Tham số mặc định (α=0.4, β=0.3, γ=0.3, δ=0.05) có thể chỉnh trong Config.

BERTScore được cache theo batch để tránh tính lại nhiều lần trong 1 epoch.
"""

import re
import torch
import numpy as np
from rouge_score import rouge_scorer as rouge_lib

try:
    from bert_score import score as _bert_score
    _BERTSCORE_AVAILABLE = True
except ImportError:
    _BERTSCORE_AVAILABLE = False
    print("⚠️  bert_score chưa cài — BERTScore reward bị tắt (chỉ dùng EM + ROUGE-L)")


# ─────────────────────────────────────────────────────────────────────────────
def _normalize(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r'[^\w\s]', '', text)
    return text


def _exact_match(pred: str, gt: str) -> float:
    return 1.0 if _normalize(pred) == _normalize(gt) else 0.0


def _rouge_l(pred: str, gt: str, scorer) -> float:
    try:
        return scorer.score(gt.lower(), pred.lower())['rougeL'].fmeasure
    except Exception:
        return 0.0


def _bertscore_batch(preds: list[str], gts: list[str]) -> list[float]:
    """BERTScore cho batch, trả về list F1 scores."""
    if not _BERTSCORE_AVAILABLE or not preds:
        return [0.0] * len(preds)
    try:
        _, _, F1 = _bert_score(
            preds, gts,
            lang="vi",
            model_type="bert-base-multilingual-cased",
            verbose=False,
        )
        return F1.tolist()
    except Exception as e:
        print(f"⚠️  BERTScore lỗi: {e}")
        return [0.0] * len(preds)


def _length_penalty(pred: str, ideal_min: int = 2, ideal_max: int = 20) -> float:
    """Phạt nếu câu quá ngắn (< ideal_min từ) hoặc quá dài (> ideal_max từ)."""
    n = len(pred.split())
    if n < ideal_min:
        return (ideal_min - n) * 0.1
    if n > ideal_max:
        return (n - ideal_max) * 0.02
    return 0.0


# ─────────────────────────────────────────────────────────────────────────────
class VQARewardModel:
    """
    Tính reward cho một batch (preds, gts).

    Dùng trong PPO training loop:
        rewards = reward_model.compute_batch(pred_texts, gt_texts)
        # rewards: list[float], mỗi phần tử ∈ [-0.5, 1.5] (thường)
    """

    def __init__(self,
                 alpha: float = 0.4,   # EM weight
                 beta:  float = 0.3,   # ROUGE-L weight
                 gamma: float = 0.3,   # BERTScore weight
                 delta: float = 0.05,  # length penalty weight
                 use_bertscore: bool = True):
        self.alpha = alpha
        self.beta  = beta
        self.gamma = gamma if (_BERTSCORE_AVAILABLE and use_bertscore) else 0.0
        self.delta = delta

        # Nếu không có BERTScore, phân bổ weight sang ROUGE-L
        if self.gamma == 0.0:
            self.beta  = self.alpha + self.beta   # ROUGE-L gánh thêm
            self.alpha = 0.5
            self.beta  = 0.5

        self._rouge = rouge_lib.RougeScorer(['rougeL'], use_stemmer=False)
        print(f"✅ RewardModel: α={self.alpha:.2f} EM, β={self.beta:.2f} ROUGE-L, "
              f"γ={self.gamma:.2f} BERTScore, δ={self.delta:.2f} LenPenalty")

    def compute_batch(self,
                      preds: list[str],
                      gts:   list[str]) -> list[float]:
        """
        Trả về reward list (float) cho mỗi cặp (pred, gt).
        Reward ∈ [−1, 1] sau clamp.
        """
        assert len(preds) == len(gts), "preds và gts phải cùng độ dài"

        em_scores    = [_exact_match(p, g)          for p, g in zip(preds, gts)]
        rouge_scores = [_rouge_l(p, g, self._rouge) for p, g in zip(preds, gts)]
        len_penalties = [_length_penalty(p)         for p in preds]

        if self.gamma > 0:
            bert_scores = _bertscore_batch(preds, gts)
        else:
            bert_scores = [0.0] * len(preds)

        rewards = []
        for em, rl, bs, lp in zip(em_scores, rouge_scores, bert_scores, len_penalties):
            r = self.alpha * em + self.beta * rl + self.gamma * bs - self.delta * lp
            r = max(-1.0, min(1.5, r))   # clamp
            rewards.append(r)

        return rewards

    def compute_single(self, pred: str, gt: str) -> float:
        return self.compute_batch([pred], [gt])[0]

    def summary_stats(self, rewards: list[float]) -> dict:
        arr = np.array(rewards)
        return {
            'mean':   float(arr.mean()),
            'std':    float(arr.std()),
            'min':    float(arr.min()),
            'max':    float(arr.max()),
            'pos_rate': float((arr > 0).mean()),
        }
