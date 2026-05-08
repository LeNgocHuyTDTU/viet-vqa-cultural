import re
import torch
import numpy as np
from tabulate import tabulate
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import single_meteor_score
from rouge_score import rouge_scorer
from bert_score import score as bert_score_func
import nltk


def normalize_vqa_text(text):
    """Làm sạch văn bản: in thường, bỏ dấu câu."""
    text = str(text).lower().strip()
    text = re.sub(r'[^\w\s]', '', text)
    return text

def exact_match(pred, gt):
    """Hàm bổ trợ để trả về 1.0 nếu khớp hoàn toàn, 0.0 nếu khác."""
    return 1.0 if normalize_vqa_text(pred) == normalize_vqa_text(gt) else 0.0

def calculate_vqa_accuracy(predictions, ground_truths):
    """Tính Accuracy trung bình cho toàn bộ tập dữ liệu."""
    if not predictions: return 0.0
    correct = sum([exact_match(p, g) for p, g in zip(predictions, ground_truths)])
    return (correct / len(predictions)) * 100

def calculate_ngram_metrics(predictions, ground_truths):
    """Tính BLEU, ROUGE-L và METEOR."""
    smoothie = SmoothingFunction().method4
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)
    
    results = {"bleu": [], "rougeL": [], "meteor": []}
    for p, g in zip(predictions, ground_truths):
        p_tokens = p.lower().split()
        g_tokens = g.lower().split()
        
        results["bleu"].append(sentence_bleu([g_tokens], p_tokens, smoothing_function=smoothie))
        results["rougeL"].append(scorer.score(g.lower(), p.lower())['rougeL'].fmeasure)
        results["meteor"].append(single_meteor_score(g_tokens, p_tokens))
        
    return {k: np.mean(v) for k, v in results.items()}

def calculate_bertscore(predictions, ground_truths):
    """Tính BERTScore ngữ nghĩa."""
    try:
        P, R, F1 = bert_score_func(
            predictions, ground_truths, lang="vi", 
            model_type="bert-base-multilingual-cased", verbose=False
        )
        return F1.mean().item()
    except Exception as e:
        print(f"⚠️ BERTScore tính toán thất bại: {e}")
        print("   Trả về 0.0 và tiếp tục...")
        return 0.0

def run_full_evaluation(predictions, ground_truths, model_label="VQA Model"):
    """Thực hiện đo lường tổng lực và in bảng kết quả."""
    if not predictions:
        print("⚠️ Không có dữ liệu dự đoán để đánh giá.")
        return {'vqa_acc': 0, 'bleu': 0, 'rougeL': 0, 'meteor': 0, 'bertscore': 0}

    metrics = {
        'vqa_acc': calculate_vqa_accuracy(predictions, ground_truths),
        'bertscore': calculate_bertscore(predictions, ground_truths)
    }
    metrics.update(calculate_ngram_metrics(predictions, ground_truths))
    
    # In bảng kết quả
    table = [
        ["VQA Accuracy (EM)", f"{metrics['vqa_acc']:.2f}%"],
        ["BLEU Score", f"{metrics['bleu']:.4f}"],
        ["ROUGE-L", f"{metrics['rougeL']:.4f}"],
        ["METEOR", f"{metrics['meteor']:.4f}"],
        ["BERTScore (F1)", f"{metrics['bertscore']:.4f}"]
    ]
    print(f"\n📊 BẢNG ĐÁNH GIÁ: {model_label.upper()}")
    print(tabulate(table, headers=["Metrics", "Value"], tablefmt="fancy_grid"))
    
    return metrics