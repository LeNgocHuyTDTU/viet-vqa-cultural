# Vietnamese Cultural VQA (Visual Question Answering)

Hệ thống hỏi-đáp trên hình ảnh với kiến thức văn hóa Việt Nam, sử dụng mô hình Vision-Language Transformer.

## 📋 Mô Tả Dự Án

Dự án xây dựng các mô hình AI để trả lời các câu hỏi về hình ảnh liên quan đến văn hóa, ẩm thực, kiến trúc, trang phục, và các khía cạnh khác của Việt Nam.

**Hướng tiếp cận hai chiều:**
- **Hướng A (Generative)**: Tạo câu trả lời tự do cho các câu hỏi phức tạp
- **Hướng B (Classification)**: Phân loại câu trả lời cho các câu hỏi đơn giản (≤15 từ)

## 🏗️ Cấu Trúc Dự Án

```
vqa_project/
├── README.md                          # File này
├── config.py                          # Cấu hình toàn cục
├── dataset.py                         # Định nghĩa Dataset classes
├── models_A.py                        # Models cho Hướng A (Generative)
├── models_B.py                        # Models cho Hướng B (ViLT)
├── train_A.py                         # Script training Hướng A
├── train_B.py                         # Script training Hướng B (ViLT + LoRA)
├── train.py                           # Training chính với Trainer API
├── evaluation.py                      # Hàm đánh giá mô hình
├── check_data.py                      # Kiểm tra tính hợp lệ dữ liệu
├── process_splits.py                  # Xử lý train/val/test splits
├── utils.py                           # Hàm tiện ích
├── app.py                             # Ứng dụng Gradio/demo
│
├── Data/
│   ├── annotations/
│   │   ├── dataset_statistics.json           # Thống kê dataset
│   │   └── vietnamese_vqa_dataset.json       # Annotation chính
│   ├── cultural_kb/
│   │   └── vietnamese_cultural_knowledge.json# Cơ sở kiến thức
│   ├── images/                               # Hình ảnh theo danh mục
│   │   ├── am_thuc/                         # Ẩm thực
│   └── splits/
│       ├── train_data.json                  # Training split
│       ├── val_data.json                    # Validation split
│       └── test_data.json                   # Test split
│
├── checkpoints/
│   ├── best_model_A1.pt                    # Model A generative (checkpoint 1)
│   ├── best_model_A2.pt                    # Model A generative (checkpoint 2)
│   ├── best_model_B1_report.json           # Report cho B1
│   ├── best_model_B1/                      # Model B classification (v1)
│   ├── best_model_B2/                      # Model B ViLT + LoRA (v2)
│   └── cache/
│       ├── train_preprocessed.pt           # Cache train data
│       └── val_preprocessed.pt             # Cache val data
```

## 🚀 Cài Đặt

### Yêu Cầu
- Python 3.8+
- GPU CUDA (khuyến nghị)
- 16GB+ RAM

### Bước 1: Clone & Setup

```bash
cd vqa_project
pip install -r requirements.txt
```

### Bước 2: Chuẩn Bị Dữ Liệu

```bash
# Kiểm tra dữ liệu
python check_data.py

# Xử lý splits
python process_splits.py
```

## 📚 Cách Sử Dụng

### Hướng A - Generative (Câu trả lời tự do)

```bash
python train_A.py
```

**Đặc điểm:**
- Mô hình: Vision Encoder + Text Decoder (BERT/GPT)
- Đầu ra: Câu trả lời tự do (không giới hạn độ dài)
- Phù hợp: Câu hỏi phân tích, so sánh, giải thích

### Hướng B - Classification (Phân loại đáp án)

#### B1: ViLT Classification cơ bản
```bash
python train_B.py
# Tự động load từ cache nếu có
```

#### B2: ViLT + LoRA Fine-tuning (Khuyến nghị)
```bash
python train_B.py --use_lora
```

**Đặc điểm:**
- Mô hình: ViLT (Vision-and-Language Transformer)
- LoRA: Peft - giảm 94% tham số trainable
- Đầu ra: Phân loại đáp án từ vocabulary
- Phù hợp: Câu hỏi yes/no, multiple-choice, đáp án ngắn (≤15 từ)

### Training Chính (Trainer API)

```bash
python train.py --model_type B --epochs 10 --batch_size 8
```

### Demo/Ứng Dụng

```bash
python app.py
# Truy cập http://localhost:7860
```

## ⚙️ Cấu Hình

Chỉnh sửa `config.py`:

```python
class Config:
    # Model
    VLM_MODEL_ID = "dandelin/vilt-b32-finetuned-vqa"
    TEXT_ENCODER = "vinai/phobert-base"
    
    # Dataset
    VILT_MAX_ANSWER_WORDS = 15  # Max words per answer
    MAX_SEQ_LENGTH = 128
    MAX_ANSWER_LENGTH = 30
    
    # Training
    EPOCHS = 10
    BATCH_SIZE_B = 8
    BATCH_SIZE_A = 4
    LEARNING_RATE = 5e-4
    
    # Paths
    TRAIN_JSON = "Data/splits/train_data.json"
    VAL_JSON = "Data/splits/val_data.json"
    TEST_JSON = "Data/splits/test_data.json"
    IMAGE_DIR = "Data/images"
    CHECKPOINT_DIR = "checkpoints"
```

## 📊 Dataset

### Thống Kê

- **Tổng samples**: ~10,000+
- **Train**: 80% (~8,000)
- **Val**: 10% (~1,000)
- **Test**: 10% (~1,000)
- **Danh mục**: 12 categories
- **Câu hỏi/ảnh**: 2-5 câu hỏi/ảnh

### Định Dạng JSON

```json
[
  {
    "image_path": "Data/images/am_thuc/banh_gio/000001.jpg",
    "category": "am_thuc",
    "questions": [
      {
        "question": "Đây là cái gì?",
        "answer": "bánh giò"
      },
      {
        "question": "Nó có nguồn gốc từ đâu?",
        "answer": "Miền Bắc Việt Nam"
      }
    ]
  }
]
```

## 🧠 Mô Hình

### Hướng A (Generative)

| Model | Encoder | Decoder | Output |
|-------|---------|---------|--------|
| A1 | CLIP/BLIP | GPT-2 Tiếng Việt | Text tự do |
| A2 | ViT | BERT Decoder | Text tự do |

### Hướng B (Classification)

| Model | Architecture | Params | Trainable |
|-------|--------------|--------|-----------|
| B1 | ViLT Base | 124.8M | 7.6M (6%) |
| B2 | ViLT + LoRA | 124.8M | 294K (0.2%) |

## 📈 Training & Evaluation

### Metrics

- **Classification (B)**: Accuracy, Precision, Recall, F1
- **Generative (A)**: BLEU, METEOR, CIDEr, SPICE

### Lệnh Evaluate

```bash
python evaluation.py --model_path checkpoints/best_model_B2
```

## 🔍 Troubleshooting

### Lỗi Collate Batch
```
RuntimeError: stack expects each tensor to be equal size
```
**Giải pháp**: Đảm bảo `image.resize()` thay vì `thumbnail()` trong preprocessing.

### Lỗi Label Size Mismatch
```
ValueError: Target size must be the same as input size
```
**Giải pháp**: Dùng `CrossEntropyLoss` thay vì `binary_cross_entropy` cho classification.

### Out of Memory
```bash
# Giảm batch size
BATCH_SIZE_B = 4

# Hoặc dùng gradient accumulation
GRADIENT_ACCUMULATION_STEPS = 2
```

## 📖 Tham Khảo

- [ViLT Paper](https://arxiv.org/abs/2102.03334)
- [PEFT (LoRA)](https://github.com/huggingface/peft)
- [Vietnamese VQA Dataset](https://huggingface.co/Dangindev/viet-cultural-vqa)

## 📝 Ghi Chú Quan Trọng

1. **Cache Preprocessing**: Script tự động cache dữ liệu sau lần preprocessing đầu tiên. Xóa `checkpoints/cache/` để tạo lại.

2. **LoRA Training**: Chỉ 0.2% tham số được training, giúp tiết kiệm bộ nhớ và thời gian.

3. **Splits**: Dữ liệu được chia theo danh mục (có thể thay đổi tỷ lệ trong `process_splits.py`).

4. **Model Selection**:
   - Chọn **Hướng A** nếu cần câu trả lời chi tiết, phức tạp
   - Chọn **Hướng B** nếu cần độ chính xác cao, answer ngắn

