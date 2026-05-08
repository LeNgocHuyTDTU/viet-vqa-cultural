import json
import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import AutoTokenizer


def normalize_image_path(raw_path: str) -> str:
    """
    Chuẩn hóa đường dẫn ảnh từ JSON sang đường dẫn thực tế trên disk.
    
    JSON lưu: "data/images/am_thuc/banh_gio/000029.jpg"  (chữ thường)
    Disk có:  "Data/images/am_thuc/banh_gio/000029.jpg"  (chữ hoa D)
    """
    if raw_path.startswith("data/"):
        raw_path = "Data/" + raw_path[5:]
    elif raw_path.startswith("DATA/"):
        raw_path = "Data/" + raw_path[5:]
    return raw_path


class VietVQADataset(Dataset):
    """
    Dataset cho Hướng A (generative): câu hỏi → câu trả lời tự do.
    
    FIX:
    - Đọc đúng cấu trúc JSON thực tế: mỗi item là 1 ảnh với nhiều câu hỏi,
      mỗi câu hỏi có field 'question' và 'answer' (không phải 'answers').
    - Chuẩn hóa đường dẫn ảnh (data/ → Data/).
    - MAX_ANSWER_LENGTH tăng lên 30 để không truncate câu trả lời dài.
    - Xử lý ảnh RGBA/Palette đúng cách.
    """

    def __init__(self, json_path: str, config, transform=None):
        self.config    = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.TEXT_ENCODER)

        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225]
            )
        ])

        # Đọc và làm phẳng JSON
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Hỗ trợ cả: JSON là list các item, hoặc JSON là 1 item đơn
        if isinstance(data, dict):
            data = [data]

        self.samples = []
        skipped = 0
        for item in data:
            # Chỉ lấy dữ liệu ẩm thực
            if item.get('category') != 'am_thuc':
                continue

            img_path = normalize_image_path(item.get('image_path', ''))

            for q in item.get('questions', []):
                question = q.get('question', '').strip()
                answer   = q.get('answer',   '').strip()

                # Bỏ qua sample rỗng
                if not question or not answer:
                    skipped += 1
                    continue

                self.samples.append({
                    'image_path': img_path,
                    'question':   question,
                    'answer':     answer
                })

        if skipped > 0:
            print(f"⚠️ Đã bỏ qua {skipped} sample rỗng.")
        print(f"✅ Dataset loaded: {len(self.samples)} samples từ {json_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]

        # ── 1. Load ảnh ────────────────────────────────────────────────────
        img_full_path = os.path.join(self.config.IMAGE_DIR, sample['image_path'])
        try:
            image = Image.open(img_full_path)
            # Xử lý ảnh có độ trong suốt (RGBA, Palette)
            if image.mode in ('P', 'RGBA'):
                image = image.convert('RGBA')
            image = image.convert('RGB')
            image = self.transform(image)
        except Exception as e:
            print(f"⚠️ Lỗi load ảnh: {img_full_path} | {e}")
            image = torch.zeros((3, 224, 224))

        # ── 2. Tokenize câu hỏi ────────────────────────────────────────────
        encoded_q = self.tokenizer(
            sample['question'],
            padding='max_length',
            truncation=True,
            max_length=self.config.MAX_SEQ_LENGTH,
            return_tensors='pt'
        )

        # ── 3. Tokenize câu trả lời ────────────────────────────────────────
        # FIX: Dùng MAX_ANSWER_LENGTH=30 thay vì 15 để tránh truncate
        encoded_a = self.tokenizer(
            sample['answer'],
            padding='max_length',
            truncation=True,
            max_length=self.config.MAX_ANSWER_LENGTH,
            return_tensors='pt'
        )

        return {
            'image':         image,
            'question_ids':  encoded_q['input_ids'].squeeze(0),
            'question_mask': encoded_q['attention_mask'].squeeze(0),
            'answer_ids':    encoded_a['input_ids'].squeeze(0),
            # Giữ lại text gốc để decode khi evaluate
            'answer_text':   sample['answer'],
            'question_text': sample['question']
        }


class VietVQADataset_ViLT(Dataset):
    """
    Dataset riêng cho Hướng B (ViLT classification).
    
    FIX: ViLT chỉ phù hợp với answer ngắn (classification).
    Tự động lọc chỉ lấy các câu hỏi có answer ≤ VILT_MAX_ANSWER_WORDS từ.
    Các câu hỏi dạng phân tích/so sánh (answer dài) → để Hướng A xử lý.
    """

    def __init__(self, json_path: str, config, label2id: dict):
        self.config   = config
        self.label2id = label2id

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if isinstance(data, dict):
            data = [data]

        self.samples  = []
        skipped_long  = 0
        skipped_oov   = 0

        for item in data:
            if item.get('category') != 'am_thuc':
                continue

            img_path = normalize_image_path(item.get('image_path', ''))
            img_full = os.path.join(config.IMAGE_DIR, img_path)

            for q in item.get('questions', []):
                question = q.get('question', '').strip()
                answer   = str(q.get('answer', '')).lower().strip()

                if not question or not answer:
                    continue

                # Lọc answer dài — không phù hợp với ViLT classification
                word_count = len(answer.split())
                if word_count > config.VILT_MAX_ANSWER_WORDS:
                    skipped_long += 1
                    continue

                # Bỏ qua answer không trong vocab (tránh KeyError khi collate)
                if answer not in label2id:
                    skipped_oov += 1
                    continue

                self.samples.append({
                    'image_path': img_full,
                    'question':   question,
                    'answer':     answer
                })

        print(f"✅ ViLT Dataset: {len(self.samples)} samples")
        if skipped_long > 0:
            print(f"Bỏ {skipped_long} sample có answer dài (>{config.VILT_MAX_ANSWER_WORDS} từ)")
        if skipped_oov > 0:
            print(f"Bỏ {skipped_oov} sample có answer ngoài vocab")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        return self.samples[idx]