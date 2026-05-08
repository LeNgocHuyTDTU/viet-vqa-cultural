import torch


class Config:
    # ── 1. Đường dẫn dữ liệu ────────────────────────────────────────────────
    TRAIN_JSON = "Data/splits_cuisine/train_data.json"
    VAL_JSON   = "Data/splits_cuisine/val_data.json"
    TEST_JSON  = "Data/splits_cuisine/test_data.json"

    IMAGE_DIR      = "."          # Đường dẫn trong JSON đã có "Data/images/..."
    CHECKPOINT_DIR = "checkpoints/"

    # ── 2. Cấu hình Hướng A ──────────────────────────────────────────────────
    DECODER_TYPE: str = "transformer"   # "lstm" hoặc "transformer"
    IMAGE_ENCODER    = "resnet50"
    TEXT_ENCODER     = "vinai/phobert-base"

    MAX_SEQ_LENGTH = 30     # Độ dài câu hỏi (token)
    MAX_Q_LEN      = MAX_SEQ_LENGTH  # Alias tương thích ngược

    # FIX: Tăng từ 15 → 30 để không bị truncate câu trả lời dài
    # Phân tích dữ liệu thực tế: answer dài nhất ~12 từ × 2 token/từ + 2 = ~26 token
    MAX_ANSWER_LENGTH = 30

    HIDDEN_DIM = 256   # Giữ 256 để tiết kiệm VRAM 4GB

    # ── 3. Cấu hình Hướng B (ViLT) ──────────────────────────────────────────
    VLM_MODEL_ID = "dandelin/vilt-b32-finetuned-vqa"

    # FIX: ViLT là classification model — chỉ phù hợp với answer ngắn (≤ 4 từ).
    # Các answer dài (câu giải thích) nên được xử lý ở Hướng A (generative).
    # Với Hướng B: lọc chỉ lấy question_type "identification" (answer ngắn).
    VILT_MAX_ANSWER_WORDS = 15   # Ngưỡng lọc: answer ngắn hơn ngưỡng này → dùng ViLT

    # ── 4. Tham số Huấn luyện ───────────────────────────────────────────────
    BATCH_SIZE_A  = 8    # Hướng A
    BATCH_SIZE_B  = 8    # ViLT nhẹ hơn PaliGemma, batch 8 ổn với 4GB VRAM
    GRAD_ACCUM_B  = 2    # Tích lũy gradient cho Hướng B
    EPOCHS        = 10
    PATIENCE      = 2
    DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── 5. Decoding Strategy (Hướng A - generate()) ─────────────────────────
    # Không dùng cho ViLT (classification, không generate)
    DECODING_STRATEGY = {
        "max_new_tokens":    15,
        "do_sample":         True,
        "temperature":       0.2,
        "top_k":             40,
        "top_p":             0.9,
        "repetition_penalty": 1.15
    }