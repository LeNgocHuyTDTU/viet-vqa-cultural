"""
dataset_rl.py
=============
Dataset classes cho RL training:
  - PreferenceDataset : DPO — cặp (chosen, rejected) tokenized
  - PPOVQADataset     : PPO — câu hỏi + ảnh, reward tính sau khi generate

Dùng chung tokenizer PhoBERT từ config.TEXT_ENCODER.
"""

import json
import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import AutoTokenizer


def _img_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]),
    ])


def _load_image(path: str, transform) -> torch.Tensor:
    """Load ảnh an toàn, trả về zeros nếu lỗi."""
    # Chuẩn hoá tiền tố
    if path.startswith("data/"):
        path = "Data/" + path[5:]
    elif path.startswith("DATA/"):
        path = "Data/" + path[5:]
    try:
        img = Image.open(path)
        if img.mode in ('P', 'RGBA'):
            img = img.convert('RGBA')
        img = img.convert('RGB')
        return transform(img)
    except Exception:
        return torch.zeros(3, 224, 224)


# ─────────────────────────────────────────────────────────────────────────────
class PreferenceDataset(Dataset):
    """
    Dataset cho DPO.

    Mỗi sample trả về:
      image            (3,224,224)
      question_ids     (L,)
      question_mask    (L,)
      chosen_ids       (T,)   — tokenized câu trả lời tốt
      rejected_ids     (T,)   — tokenized câu trả lời tệ
      chosen_text      str
      rejected_text    str
    """

    def __init__(self, json_path: str, config):
        self.config    = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.TEXT_ENCODER)
        self.transform = _img_transform()

        with open(json_path, 'r', encoding='utf-8') as f:
            self.pairs = json.load(f)

        print(f"✅ PreferenceDataset: {len(self.pairs)} cặp từ {json_path}")

    def __len__(self):
        return len(self.pairs)

    def _tok(self, text: str, max_len: int) -> dict:
        return self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=max_len,
            return_tensors='pt',
        )

    def __getitem__(self, idx: int) -> dict:
        p = self.pairs[idx]

        image = _load_image(p['image_path'], self.transform)

        eq = self._tok(p['question'], self.config.MAX_SEQ_LENGTH)
        ec = self._tok(p['chosen'],   self.config.MAX_ANSWER_LENGTH)
        er = self._tok(p['rejected'], self.config.MAX_ANSWER_LENGTH)

        return {
            'image':          image,
            'question_ids':   eq['input_ids'].squeeze(0),
            'question_mask':  eq['attention_mask'].squeeze(0),
            'chosen_ids':     ec['input_ids'].squeeze(0),
            'rejected_ids':   er['input_ids'].squeeze(0),
            'chosen_text':    p['chosen'],
            'rejected_text':  p['rejected'],
            'question_text':  p['question'],
            'question_type':  p.get('question_type', ''),
        }


# ─────────────────────────────────────────────────────────────────────────────
class PPOVQADataset(Dataset):
    """
    Dataset cho PPO: chỉ cần (image, question).
    Reward được tính sau khi model generate() ra câu trả lời.

    ground_truth_answer được trả về để tính reward (VQA accuracy / BERTScore).
    """

    def __init__(self, json_path: str, config):
        self.config    = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.TEXT_ENCODER)
        self.transform = _img_transform()

        with open(json_path, 'r', encoding='utf-8') as f:
            pairs = json.load(f)

        # Chỉ dùng chosen làm ground truth
        self.samples = [
            {
                'image_path':  p['image_path'],
                'question':    p['question'],
                'answer':      p['chosen'],      # ground truth
                'question_type': p.get('question_type', ''),
            }
            for p in pairs
        ]
        print(f"✅ PPOVQADataset: {len(self.samples)} samples từ {json_path}")

    def __len__(self):
        return len(self.samples)

    def _tok(self, text: str, max_len: int) -> dict:
        return self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=max_len,
            return_tensors='pt',
        )

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]

        image = _load_image(s['image_path'], self.transform)
        eq    = self._tok(s['question'], self.config.MAX_SEQ_LENGTH)

        return {
            'image':         image,
            'question_ids':  eq['input_ids'].squeeze(0),
            'question_mask': eq['attention_mask'].squeeze(0),
            'answer_text':   s['answer'],
            'question_text': s['question'],
            'question_type': s['question_type'],
        }
