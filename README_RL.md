# RL Training Pipeline — VietVQA Ẩm Thực
# =========================================
# Hướng dẫn chạy toàn bộ pipeline RL (DPO + PPO) và so sánh với SFT

## Cấu trúc file

```
├── build_preference_data.py   # Bước 0: Tạo preference_data.json từ test_data.json
├── dataset_rl.py              # Dataset cho DPO (PreferenceDataset) và PPO (PPOVQADataset)
├── reward_model.py            # Reward function: α·EM + β·ROUGE-L + γ·BERTScore
├── train_dpo.py               # Bước 1A: DPO training
├── train_ppo.py               # Bước 1B: PPO (REINFORCE + KL) training
├── evaluate_rl_vs_sft.py      # Bước 2: So sánh SFT vs DPO vs PPO
├── analyze_human_eval.py      # Bước 3: Phân tích human evaluation
└── preference_data.json       # 2166 cặp chosen/rejected (auto-generated)
```

## Pipeline 4 bước

### Bước 0 — Tạo Preference Data
```bash
# Từ test_data.json thực tế (442 ảnh × 5 câu hỏi = 2166 cặp)
python build_preference_data.py
# → preference_data.json (2166 cặp, >> yêu cầu tối thiểu 100 cặp)
```

### Bước 1A — Train DPO
```bash
python train_dpo.py --sft_checkpoint checkpoints/best_model_A_transformer.pt
# → checkpoints/best_model_dpo.pt
# → dpo_training_history.json
```

### Bước 1B — Train PPO
```bash
python train_ppo.py --sft_checkpoint checkpoints/best_model_A_transformer.pt
# → checkpoints/best_model_ppo.pt
# → ppo_training_history.json
```

*DPO và PPO có thể chạy song song nếu đủ VRAM.*

### Bước 2 — Đánh giá tự động
```bash
python evaluate_rl_vs_sft.py --max_samples 500
# In bảng so sánh SFT vs DPO vs PPO:
#   - VQA Accuracy, BLEU, ROUGE-L, METEOR, BERTScore, Reward
#   - Breakdown theo question_type
# → human_eval_sheet.json (30 mẫu để người đánh giá)
# → rl_vs_sft_results.json (kết quả đầy đủ)
```

### Bước 3 — Human Evaluation
```bash
# 1. Mở human_eval_sheet.json, điền điểm cho từng response:
#    correctness_1_5, fluency_1_5, cultural_rel_1_5 (thang 1-5)
#    overall_preference: "SFT" | "DPO" | "PPO"

# 2. Phân tích kết quả
python analyze_human_eval.py
# In bảng:
#   - Điểm trung bình (Correctness / Fluency / Cultural Relevance)
#   - Win/Loss/Tie matrix
#   - Tổng kết RL vs SFT
```

---

## Thiết kế Preference Data (2166 cặp)

| Loại lỗi trong Rejected | Tỷ lệ | Ví dụ |
|--------------------------|-------|-------|
| Sai tên món ăn           | ~30%  | "Đây là bún bò Huế" thay vì "Cơm âm phủ Huế" |
| Quá mơ hồ/chung chung    | ~25%  | "Đây là một món ăn Việt Nam" |
| Sai ngôn ngữ (English)   | ~15%  | "This is a Vietnamese dish" |
| Thiếu thông tin (truncated) | ~15% | Chỉ lấy phần đầu của câu trả lời |
| Nhầm vùng miền           | ~15%  | "Phản ánh ẩm thực Hà Nội" thay vì "Huế" |

Phân phối theo question_type:
- identification: 442  (easy)
- description:    430  (medium)
- cultural:       437  (medium)
- analysis:       419  (hard)
- comparison:     438  (hard)

---

## Thiết kế Reward Model (PPO)

```
reward = α·ExactMatch + β·ROUGE-L + γ·BERTScore − δ·LengthPenalty
       = 0.4·EM       + 0.3·RL    + 0.3·BS     − 0.05·LP
```

- **ExactMatch**: 1.0 nếu khớp hoàn toàn sau normalize
- **ROUGE-L**: F1 overlap subsequence (phù hợp câu tiếng Việt)
- **BERTScore**: cosine similarity embeddings multilingual BERT
- **LengthPenalty**: phạt câu < 2 từ hoặc > 20 từ

---

## DPO Loss

```
L_DPO = -E[log σ(β·(log π_θ(chosen|x) - log π_ref(chosen|x))
                - β·(log π_θ(rejected|x) - log π_ref(rejected|x)))]
```
- β = 0.1 (KL coefficient)
- Không cần reward model riêng
- Ổn định hơn PPO cho model nhỏ

---

## PPO Loss (REINFORCE + KL variant)

```
L = -E[(R - baseline) · log π(a|s)] + λ_kl · KL(π || π_ref)
```
- R = VQARewardModel score
- baseline = mean(R) trong batch (giảm variance)
- λ_kl = 0.05 (tránh model drift quá xa SFT)

---

## Yêu cầu cài đặt

```bash
pip install torch transformers torchvision
pip install rouge-score bert-score nltk tabulate
```

---

## Lưu ý VRAM (4GB)

| Model        | Batch Size | Approx VRAM |
|--------------|-----------|-------------|
| DPO train    | 8         | ~3.5 GB     |
| PPO rollout  | 4         | ~3.0 GB     |
| PPO update   | 4         | ~3.5 GB     |
| Evaluate     | 8         | ~2.5 GB     |

Nếu OOM: giảm BATCH_SIZE_A = 4 trong config.py
