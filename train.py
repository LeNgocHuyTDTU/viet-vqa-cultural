import json
import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import AutoTokenizer

class VietVQADataset(Dataset):
    def __init__(self, json_path, config, transform=None):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.TEXT_ENCODER)
        
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Đọc và làm phẳng file JSON
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        self.samples = []
        for item in data:
            img_path = item.get('image_path', '')
            for q in item.get('questions', []):
                self.samples.append({
                    'image_path': img_path,
                    'question': q['question'],
                    'answer': q['answer']
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 1. Load Ảnh
        img_full_path = os.path.join(self.config.IMAGE_DIR, sample['image_path'])
        try:
            image = Image.open(img_full_path).convert('RGB')
            image = self.transform(image)
        except Exception:
            image = torch.zeros((3, 224, 224)) # Dummy nếu lỗi

        # 2. Tokenize Câu hỏi
        encoded_q = self.tokenizer(
            sample['question'],
            padding='max_length',
            truncation=True,
            max_length=self.config.MAX_SEQ_LENGTH,
            return_tensors='pt'
        )
        
        # 3. Tokenize Câu trả lời
        encoded_a = self.tokenizer(
            sample['answer'],
            padding='max_length',
            truncation=True,
            max_length=self.config.MAX_ANSWER_LENGTH,
            return_tensors='pt'
        )

        return {
            'image': image,
            'question_ids': encoded_q['input_ids'].squeeze(0),
            'question_mask': encoded_q['attention_mask'].squeeze(0),
            'answer_ids': encoded_a['input_ids'].squeeze(0)
        }