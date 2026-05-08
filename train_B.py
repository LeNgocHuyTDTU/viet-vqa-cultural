import os
import json
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from config import Config
from dataset import VietVQADataset_ViLT
from peft import LoraConfig, get_peft_model

from transformers import ViltProcessor, ViltForQuestionAnswering
from datasets import load_dataset


def build_vocab(config) -> tuple[dict, dict]:
    unique_answers = set()
    
    # 1. Quét dữ liệu từ file JSON cũ
    for path in [config.TRAIN_JSON, config.VAL_JSON, config.TEST_JSON]:
        if not os.path.exists(path): continue
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, dict): data = [data]
        for item in data:
            if item.get('category') != 'am_thuc': continue
            for q in item.get('questions', []):
                ans = str(q.get('answer', '')).lower().strip()
                if ans and len(ans.split()) <= config.VILT_MAX_ANSWER_WORDS:
                    unique_answers.add(ans)

    # 2. Quét thêm dữ liệu từ Hugging Face
    print("Đang tải dữ liệu từ Hugging Face để xây dựng từ điển...")
    try:
        hf_train = load_dataset("Dangindev/viet-cultural-vqa", split="train")
        for sample in hf_train:
            for q in sample.get('questions', []):
                ans = str(q.get('answer', '')).lower().strip()
                if ans and len(ans.split()) <= config.VILT_MAX_ANSWER_WORDS:
                    unique_answers.add(ans)
        print("✅ Đã gộp thành công từ vựng từ Hugging Face.")
    except Exception as e:
        print(f"⚠️ Lỗi khi tải dataset Hugging Face: {e}. Tiếp tục với dữ liệu cục bộ.")

    sorted_answers = sorted(list(unique_answers))
    id2label = {i: label for i, label in enumerate(sorted_answers)}
    label2id = {label: i for i, label in id2label.items()}
    return id2label, label2id


class VietVQADataset_HF_ViLT(Dataset):
    """Adapter chuyển đổi định dạng Hugging Face Dataset."""
    def __init__(self, hf_split, config, label2id):
        self.config = config
        self.label2id = label2id
        self.samples = []

        print("Đang map Hugging Face Dataset sang chuẩn DataLoader...")
        for item in tqdm(hf_split, desc="Processing HF"):
            img = item['image'] # PIL Image có sẵn
            for q in item.get('questions', []):
                question = str(q.get('question', '')).strip()
                answer   = str(q.get('answer', '')).lower().strip()

                if not question or not answer: continue
                if len(answer.split()) > config.VILT_MAX_ANSWER_WORDS: continue
                if answer not in label2id: continue

                self.samples.append({
                    'image': img,
                    'question': question,
                    'answer': answer
                })

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx: int) -> dict: return self.samples[idx]


class TokenizedVQADataset(Dataset):
    def __init__(self, samples, processor, label2id):
        self.features = []
        self.label2id = label2id

        print("Đang tiền xử lý dữ liệu theo kiểu map trước khi train...")
        for item in tqdm(samples, desc="Preprocessing"):
            try:
                if 'image_path' in item:
                    image = Image.open(item['image_path']).convert("RGB")
                else:
                    image = item['image'].convert("RGB")
                # FIX: Dùng resize() thay vì thumbnail() để đảm bảo kích thước cố định
                # thumbnail() giữ aspect ratio → kích thước khác nhau → lỗi collate
                image = image.resize((384, 384), Image.Resampling.LANCZOS)
            except Exception as e:
                print(f"⚠️ Lỗi ảnh: {e}")
                image = Image.new("RGB", (224, 224))

            encoding = processor(
                image,
                item['question'],
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            feature = {key: value.squeeze(0) for key, value in encoding.items()}
            # FIX: ViLT expects class index, not one-hot encoded label
            label_idx = label2id.get(item['answer'], -1)
            if label_idx >= 0:
                feature["labels"] = torch.tensor(label_idx, dtype=torch.long)
                self.features.append(feature)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx]


def extract_samples(dataset):
    if isinstance(dataset, Dataset):
        return [dataset[i] for i in range(len(dataset))]
    return list(dataset)


def preprocess_dataset(dataset, processor, label2id):
    samples = extract_samples(dataset)
    return TokenizedVQADataset(samples, processor, label2id)


def run_B2_finetune_raw_loop(config, processor, train_ds, val_ds, id2label, label2id):
    print("\n" + "="*60)
    print("  [B2] VILT + LORA FINE-TUNING (RAW LOOP)")
    print("="*60)
    
    # ── Load model ──────────────────────────────────────────────
    model = ViltForQuestionAnswering.from_pretrained(
        config.VLM_MODEL_ID, 
        num_labels=len(id2label),
        id2label=id2label, 
        label2id=label2id, 
        ignore_mismatched_sizes=True,
    ).to(config.DEVICE)

    # ── Apply LoRA ──────────────────────────────────────────────
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["query", "value"],
        lora_dropout=0.05,
        bias="none"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_dataloader = DataLoader(
        train_ds, 
        batch_size=config.BATCH_SIZE_B, 
        shuffle=True,
        num_workers=0,  # FIX: Set to 0 để tránh issues
        pin_memory=True
    )

    val_dataloader = DataLoader(
        val_ds,
        batch_size=config.BATCH_SIZE_B,
        shuffle=False,
        num_workers=0,  # FIX: Set to 0 để tránh issues
        pin_memory=True,
    )

    # ── Optimizer & Scheduler ───────────────────────────────────
    optimizer = AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
    total_steps = len(train_dataloader) * config.EPOCHS
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-5)
    
    scaler = torch.amp.GradScaler('cuda')
    criterion = nn.CrossEntropyLoss()

    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(config.EPOCHS):
        # ════════════════════════════════════════════════════════════════
        # TRAINING PHASE
        # ════════════════════════════════════════════════════════════════
        model.train()
        total_loss = 0
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{config.EPOCHS} [Train]")
        
        for step, batch in enumerate(pbar):
            batch = {k: v.to(config.DEVICE) if torch.is_tensor(v) else v 
                     for k, v in batch.items()}
            
            # FIX: ViLT VQA mode expects multi-hot labels, but we have single class indices
            # → Compute CrossEntropyLoss manually instead
            labels = batch.pop("labels")
            
            optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast('cuda'):
                outputs = model(**batch)
                logits = outputs.logits
                loss = criterion(logits, labels)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            total_loss += loss.item()
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{optimizer.param_groups[0]['lr']:.2e}"
            })
            
        avg_train_loss = total_loss / len(train_dataloader)
        print(f"📉 Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}")

        # ════════════════════════════════════════════════════════════════
        # VALIDATION PHASE
        # ════════════════════════════════════════════════════════════════
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc=f"Epoch {epoch+1} [Val]"):
                batch = {k: v.to(config.DEVICE) if torch.is_tensor(v) else v 
                         for k, v in batch.items()}
                
                # FIX: Loại labels ra, tính loss thủ công
                labels = batch.pop("labels")
                
                with torch.amp.autocast('cuda'):
                    outputs = model(**batch)
                    logits = outputs.logits
                    loss = criterion(logits, labels)
                
                val_loss += loss.item()
                
                # Tính accuracy
                preds = logits.argmax(-1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        avg_val_loss = val_loss / max(len(val_dataloader), 1)
        val_accuracy = (val_correct / val_total * 100) if val_total > 0 else 0
        print(f"📊 Epoch {epoch+1}: Val Loss = {avg_val_loss:.4f} | Accuracy = {val_accuracy:.2f}%")

        # ════════════════════════════════════════════════════════════════
        # SAVE & EARLY STOPPING
        # ════════════════════════════════════════════════════════════════
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            save_path = os.path.join(config.CHECKPOINT_DIR, "best_model_B2")
            model.save_pretrained(save_path)
            processor.save_pretrained(save_path)
            print(f"✅ Checkpoint lưu tại: {save_path} (Loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"⏳ No improvement. Patience: {patience_counter}/{config.PATIENCE}")
            
            if patience_counter >= config.PATIENCE:
                print(f"🛑 Early stopping at epoch {epoch+1}")
                break

    print("\n✅ Training hoàn thành!")


if __name__ == "__main__":
    cfg = Config()
    
    print("1. Xây dựng từ điển answer ngắn cho ViLT...")
    id2label, label2id = build_vocab(cfg)
    print(f"✅ Đã chốt {len(id2label)} nhãn (answer ≤ {cfg.VILT_MAX_ANSWER_WORDS} từ).")
    
    proc = ViltProcessor.from_pretrained(cfg.VLM_MODEL_ID)
    
    # ── Cache paths ─────────────────────────────────────────────
    cache_dir = os.path.join(cfg.CHECKPOINT_DIR, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    train_cache = os.path.join(cache_dir, "train_preprocessed.pt")
    val_cache = os.path.join(cache_dir, "val_preprocessed.pt")
    
    print("\n2. Load dữ liệu Cục bộ (JSON)...")
    train_data_local = VietVQADataset_ViLT(cfg.TRAIN_JSON, cfg, label2id)
    val_data_local   = VietVQADataset_ViLT(cfg.VAL_JSON,   cfg, label2id)

    print("\n3. Tiền xử lý dữ liệu theo kiểu map...")
    
    # ── Kiểm tra cache ──────────────────────────────────────────
    if os.path.exists(train_cache) and os.path.exists(val_cache):
        print("⚡ Load dữ liệu từ cache...")
        train_data_combined = torch.load(train_cache)
        val_data_combined = torch.load(val_cache)
        print(f"✅ Cache loaded!")
    else:
        print("💾 Preprocessing + lưu cache...")
        train_data_combined = preprocess_dataset(train_data_local, proc, label2id)
        val_data_combined   = preprocess_dataset(val_data_local, proc, label2id)
        
        # Lưu cache để lần sau load nhanh
        torch.save(train_data_combined, train_cache)
        torch.save(val_data_combined, val_cache)
        print(f"💾 Cache lưu tại: {cache_dir}")
    
    print(f"🔥 Tổng mẫu Train: {len(train_data_combined)}")
    print(f"🔥 Tổng mẫu Val: {len(val_data_combined)}")
    
    if len(train_data_combined) == 0:
        print("❌ Dataset rỗng. Dừng chương trình.")
        exit(1)
        
    print("\n5. Bắt đầu Huấn luyện...")
    run_B2_finetune_raw_loop(cfg, proc, train_data_combined, val_data_combined, id2label, label2id)