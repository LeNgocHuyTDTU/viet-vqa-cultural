import os
import torch
import torch.nn as nn
from tqdm import tqdm
from config import Config
from models_A import VQA_Model_A
from dataset import VietVQADataset
from evaluation import run_full_evaluation


def train_A():
    config = Config()
    print(f"TRAINING HƯỚNG A ({config.DECODER_TYPE.upper()})")

    # ── Dataset & DataLoader ────────────────────────────────────────────────
    train_ds = VietVQADataset(config.TRAIN_JSON, config)
    val_ds   = VietVQADataset(config.VAL_JSON,   config)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=config.BATCH_SIZE_A,
        shuffle=True, num_workers=2, pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=config.BATCH_SIZE_A,
        shuffle=False, num_workers=2, pin_memory=True
    )

    tokenizer  = train_ds.tokenizer
    vocab_size = tokenizer.vocab_size

    # ── Model, Optimizer, Loss ──────────────────────────────────────────────
    model     = VQA_Model_A(config, vocab_size).to(config.DEVICE)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-4, weight_decay=1e-2
    )
    # Pad token không đóng góp vào loss
    # Tại train_A.py
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, label_smoothing=0.1)
    scaler    = torch.amp.GradScaler('cuda')

    # ── Token đặc biệt cho Generate ────────────────────────────────────────
    bos_id = tokenizer.cls_token_id or tokenizer.bos_token_id or 0
    eos_id = tokenizer.sep_token_id or tokenizer.eos_token_id or 2

    # ── Best metrics tracker ────────────────────────────────────────────────
    best_metrics = {
        'val_loss': float('inf'),
        'vqa_acc': 0.0, 'bleu': 0.0, 'rougeL': 0.0, 'meteor': 0.0, 'bertscore': 0.0
    }
    patience_counter = 0
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    for epoch in range(config.EPOCHS):

        # ════════════════════════════════════════════════════════════════════
        # PHASE 1: TRAIN (Teacher Forcing)
        # ════════════════════════════════════════════════════════════════════
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS} [Train]")

        for batch in pbar:
            imgs    = batch['image'].to(config.DEVICE)
            q_ids   = batch['question_ids'].to(config.DEVICE)
            q_mask  = batch['question_mask'].to(config.DEVICE)
            a_ids   = batch['answer_ids'].to(config.DEVICE)   # (B, T)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda'):
                # FIX: forward() trả về (B, T, vocab) — khớp đúng với a_ids (B, T)
                logits = model(imgs, q_ids, q_mask, tgt_ids=a_ids)  # (B, T, V)

                # Shift: input = a_ids[:,:-1], label = a_ids[:,1:]
                shift_logits = logits[:, :-1, :].contiguous()        # (B, T-1, V)
                shift_labels = a_ids[:, 1:].contiguous()              # (B, T-1)

                loss = criterion(
                    shift_logits.view(-1, vocab_size),   # (B*(T-1), V)
                    shift_labels.view(-1)                 # (B*(T-1),)
                )

            scaler.scale(loss).backward()
            # Gradient clipping tránh exploding gradient
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = train_loss / len(train_loader)

        # ════════════════════════════════════════════════════════════════════
        # PHASE 2: VALIDATION
        # - Loss: tính bằng teacher forcing (nhanh, ổn định)
        # - Metrics: generate() thật sự để đo chất lượng sinh câu
        # ════════════════════════════════════════════════════════════════════
        model.eval()
        val_loss_sum = 0.0
        all_preds, all_gts = [], []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS} [Val]"):
                imgs   = batch['image'].to(config.DEVICE)
                q_ids  = batch['question_ids'].to(config.DEVICE)
                q_mask = batch['question_mask'].to(config.DEVICE)
                a_ids  = batch['answer_ids'].to(config.DEVICE)

                # ── Val Loss (teacher forcing) ──────────────────────────
                with torch.amp.autocast('cuda'):
                    logits = model(imgs, q_ids, q_mask, tgt_ids=a_ids)
                    shift_logits = logits[:, :-1, :].contiguous()
                    shift_labels = a_ids[:, 1:].contiguous()
                    loss = criterion(shift_logits.view(-1, vocab_size), shift_labels.view(-1))
                    val_loss_sum += loss.item()

                # ── Generate thật sự (không teacher forcing) ───────────
                # FIX: Dùng model.generate() để đo metrics thực tế
                pred_token_lists = model.generate(
                    imgs, q_ids, q_mask,
                    bos_token_id=bos_id,
                    eos_token_id=eos_id,
                    max_len=config.MAX_ANSWER_LENGTH,
                    beam_size=1
                )

                pred_texts = tokenizer.batch_decode(pred_token_lists, skip_special_tokens=True)
                gt_texts   = tokenizer.batch_decode(a_ids.tolist(), skip_special_tokens=True)

                all_preds.extend(pred_texts)
                all_gts.extend(gt_texts)

        avg_val_loss = val_loss_sum / len(val_loader)
        print(f"\n📉 Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # ════════════════════════════════════════════════════════════════════
        # PHASE 3: ĐÁNH GIÁ 6 CHỈ SỐ
        # ════════════════════════════════════════════════════════════════════
        current_metrics = run_full_evaluation(
            all_preds, all_gts, model_label=f"A-{config.DECODER_TYPE.upper()} Epoch {epoch+1}"
        )
        current_metrics['val_loss'] = avg_val_loss

        # ── Regression check ────────────────────────────────────────────────
        regress_count = 0
        if current_metrics['val_loss'] > best_metrics['val_loss']:
            regress_count += 1
        for key in ['vqa_acc', 'bleu', 'rougeL', 'meteor', 'bertscore']:
            if current_metrics[key] < best_metrics[key]:
                regress_count += 1

        print(f"Kiểm định: {regress_count}/6 chỉ số bị sụt giảm.")

        # ── Lưu Checkpoint ──────────────────────────────────────────────────
        improved = (
            current_metrics['vqa_acc'] > best_metrics['vqa_acc'] or
            current_metrics['val_loss'] < best_metrics['val_loss']
        )
        if regress_count < 3 and improved:
            best_metrics = current_metrics.copy()
            ckpt_path = os.path.join(config.CHECKPOINT_DIR, f"best_model_A_{config.DECODER_TYPE}.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"Checkpoint lưu! VQA Acc: {current_metrics['vqa_acc']:.2f}% | Loss: {avg_val_loss:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"Không cải thiện. Early Stopping: {patience_counter}/{config.PATIENCE}")
            if patience_counter >= config.PATIENCE:
                print("Dừng sớm để tránh Overfitting.")
                break

    print(f"\nHuấn luyện hoàn tất. Best VQA Acc: {best_metrics['vqa_acc']:.2f}%")


if __name__ == "__main__":
    train_A()