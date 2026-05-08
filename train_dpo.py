"""
train_dpo.py
============
Direct Preference Optimization (DPO) cho VQA_Model_A.

Mục tiêu: dạy model sinh ra "chosen" hơn "rejected" mà không cần
          reward model riêng biệt. Loss DPO đơn giản hơn PPO và
          ổn định hơn cho seq2seq nhỏ.

DPO Loss (Rafailov et al., 2023):
  L_DPO = -E[ log σ( β·(log π_θ(chosen|x) − log π_ref(chosen|x))
                    − β·(log π_θ(rejected|x) − log π_ref(rejected|x)) ) ]

Trong đó:
  π_θ   = model đang train
  π_ref = model frozen (SFT checkpoint)
  β     = KL penalty coefficient (thường 0.1 ~ 0.5)

Pipeline:
  1. Load SFT checkpoint làm reference model (frozen)
  2. Clone thành policy model (trainable)
  3. Với mỗi batch (img, q, chosen, rejected):
     a. Tính log-prob của chosen và rejected dưới cả 2 model
     b. Tính DPO loss
     c. Backward + update policy model
  4. Validate và lưu checkpoint tốt nhất

Chạy:
  python train_dpo.py
  python train_dpo.py --sft_checkpoint checkpoints/best_model_A_transformer.pt
"""

import os
import copy
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from config import Config
from models_A import VQA_Model_A
from dataset_rl import PreferenceDataset
from evaluation import run_full_evaluation


# ─────────────────────────────────────────────────────────────────────────────
def compute_sequence_logprob(
    model: VQA_Model_A,
    images:       torch.Tensor,   # (B,3,H,W)
    q_ids:        torch.Tensor,   # (B,L)
    q_mask:       torch.Tensor,   # (B,L)
    answer_ids:   torch.Tensor,   # (B,T)
    pad_id:       int,
) -> torch.Tensor:
    """
    Tính log-probability của answer sequence dưới model.
    Trả về (B,) — tổng log-prob per sample (không tính padding).
    """
    logits = model(images, q_ids, q_mask, tgt_ids=answer_ids)  # (B,T,V)

    # Shift: predict token t+1 từ token t
    shift_logits = logits[:, :-1, :].contiguous()          # (B,T-1,V)
    shift_labels = answer_ids[:, 1:].contiguous()           # (B,T-1)

    log_probs = F.log_softmax(shift_logits, dim=-1)         # (B,T-1,V)

    # Lấy log-prob của token đúng tại mỗi bước
    # gather: (B,T-1,1) → squeeze → (B,T-1)
    token_logp = log_probs.gather(
        dim=2,
        index=shift_labels.unsqueeze(2).clamp(min=0)
    ).squeeze(2)                                            # (B,T-1)

    # Mask padding (pad_id = 1 với PhoBERT)
    mask = (shift_labels != pad_id).float()                 # (B,T-1)
    seq_logp = (token_logp * mask).sum(dim=1)               # (B,)

    return seq_logp


def dpo_loss(
    policy_chosen_logp:   torch.Tensor,  # (B,)
    policy_rejected_logp: torch.Tensor,  # (B,)
    ref_chosen_logp:      torch.Tensor,  # (B,)
    ref_rejected_logp:    torch.Tensor,  # (B,)
    beta:                 float = 0.1,
) -> tuple[torch.Tensor, dict]:
    """
    DPO loss theo paper Rafailov et al. (2023).

    Returns:
        loss   : scalar tensor
        stats  : dict với các số liệu diagnostic
    """
    # Tỷ số log-prob: policy vs reference
    pi_logratios = policy_chosen_logp   - policy_rejected_logp
    ref_logratios = ref_chosen_logp     - ref_rejected_logp

    # DPO objective
    logits_diff = beta * (pi_logratios - ref_logratios)
    loss = -F.logsigmoid(logits_diff).mean()

    # Stats diagnostic
    with torch.no_grad():
        chosen_rewards   = beta * (policy_chosen_logp   - ref_chosen_logp).detach()
        rejected_rewards = beta * (policy_rejected_logp - ref_rejected_logp).detach()
        reward_acc = (chosen_rewards > rejected_rewards).float().mean()
        reward_margin = (chosen_rewards - rejected_rewards).mean()

    stats = {
        'dpo_loss':       loss.item(),
        'reward_acc':     reward_acc.item(),
        'reward_margin':  reward_margin.item(),
        'chosen_reward':  chosen_rewards.mean().item(),
        'rejected_reward': rejected_rewards.mean().item(),
    }
    return loss, stats


# ─────────────────────────────────────────────────────────────────────────────
def train_dpo(sft_checkpoint: str | None = None):
    config = Config()

    # ── Preference Dataset ──────────────────────────────────────────────────
    full_ds = PreferenceDataset("preference_data.json", config)

    val_size   = max(1, int(0.1 * len(full_ds)))   # 10% validation
    train_size = len(full_ds) - val_size
    train_ds, val_ds = random_split(full_ds, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=8,  shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=8,  shuffle=False,
                              num_workers=2, pin_memory=True)

    tokenizer  = full_ds.tokenizer
    vocab_size = tokenizer.vocab_size
    pad_id     = tokenizer.pad_token_id or 1
    bos_id     = tokenizer.cls_token_id or 0
    eos_id     = tokenizer.sep_token_id or 2

    # ── Reference Model (frozen SFT) ────────────────────────────────────────
    ref_model = VQA_Model_A(config, vocab_size).to(config.DEVICE)
    if sft_checkpoint and os.path.exists(sft_checkpoint):
        ref_model.load_state_dict(
            torch.load(sft_checkpoint, map_location=config.DEVICE)
        )
        print(f"✅ Reference model loaded: {sft_checkpoint}")
    else:
        print("⚠️  Không tìm thấy SFT checkpoint — dùng random init làm reference.")

    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    # ── Policy Model (trainable clone) ──────────────────────────────────────
    policy_model = copy.deepcopy(ref_model).to(config.DEVICE)
    # Mở đóng băng cho decoder + projection layers
    for name, p in policy_model.named_parameters():
        if any(k in name for k in ['transformer_decoder', 'lstm', 'fc_out',
                                    'cross_attn', 'token_embedding',
                                    'img_spatial_proj', 'img_pool_proj', 'txt_proj']):
            p.requires_grad = True
        else:
            p.requires_grad = False

    trainable = sum(p.numel() for p in policy_model.parameters() if p.requires_grad)
    print(f"Trainable params (policy): {trainable:,}")

    optimizer = torch.optim.AdamW(
        [p for p in policy_model.parameters() if p.requires_grad],
        lr=5e-5, weight_decay=1e-2
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.EPOCHS
    )
    scaler = torch.amp.GradScaler('cuda')

    beta           = 0.1     # DPO KL coefficient
    best_reward_acc = 0.0
    patience_counter = 0
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    history = []

    # ════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"  DPO Training  |  β={beta}  |  {train_size} train  |  {val_size} val")
    print(f"{'='*60}\n")

    for epoch in range(config.EPOCHS):

        # ── Train ────────────────────────────────────────────────────────
        policy_model.train()
        epoch_stats = {'dpo_loss': [], 'reward_acc': [], 'reward_margin': []}

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS} [DPO Train]")
        for batch in pbar:
            imgs     = batch['image'].to(config.DEVICE)
            q_ids    = batch['question_ids'].to(config.DEVICE)
            q_mask   = batch['question_mask'].to(config.DEVICE)
            c_ids    = batch['chosen_ids'].to(config.DEVICE)
            r_ids    = batch['rejected_ids'].to(config.DEVICE)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda'):
                # Policy log-probs
                pi_chosen_logp   = compute_sequence_logprob(
                    policy_model, imgs, q_ids, q_mask, c_ids, pad_id)
                pi_rejected_logp = compute_sequence_logprob(
                    policy_model, imgs, q_ids, q_mask, r_ids, pad_id)

            # Reference log-probs (no grad)
            with torch.no_grad():
                ref_chosen_logp   = compute_sequence_logprob(
                    ref_model, imgs, q_ids, q_mask, c_ids, pad_id)
                ref_rejected_logp = compute_sequence_logprob(
                    ref_model, imgs, q_ids, q_mask, r_ids, pad_id)

            with torch.amp.autocast('cuda'):
                loss, stats = dpo_loss(
                    pi_chosen_logp, pi_rejected_logp,
                    ref_chosen_logp, ref_rejected_logp,
                    beta=beta
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            for k in epoch_stats:
                epoch_stats[k].append(stats[k])

            pbar.set_postfix(
                loss=f"{stats['dpo_loss']:.4f}",
                racc=f"{stats['reward_acc']:.3f}",
                margin=f"{stats['reward_margin']:.3f}"
            )

        scheduler.step()

        avg = {k: sum(v)/len(v) for k, v in epoch_stats.items()}
        print(f"\n📈 Epoch {epoch+1} Train — "
              f"DPO Loss: {avg['dpo_loss']:.4f} | "
              f"Reward Acc: {avg['reward_acc']:.3f} | "
              f"Margin: {avg['reward_margin']:.4f}")

        # ── Validate (generate & evaluate) ──────────────────────────────
        policy_model.eval()
        all_preds, all_gts = [], []
        val_reward_accs = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1} [DPO Val]"):
                imgs   = batch['image'].to(config.DEVICE)
                q_ids  = batch['question_ids'].to(config.DEVICE)
                q_mask = batch['question_mask'].to(config.DEVICE)
                c_ids  = batch['chosen_ids'].to(config.DEVICE)
                r_ids  = batch['rejected_ids'].to(config.DEVICE)

                # Reward accuracy trên val
                pi_c = compute_sequence_logprob(policy_model, imgs, q_ids, q_mask, c_ids, pad_id)
                pi_r = compute_sequence_logprob(policy_model, imgs, q_ids, q_mask, r_ids, pad_id)
                val_reward_accs.append((pi_c > pi_r).float().mean().item())

                # Generate cho evaluation
                preds = policy_model.generate(
                    imgs, q_ids, q_mask,
                    bos_token_id=bos_id,
                    eos_token_id=eos_id,
                    max_len=config.MAX_ANSWER_LENGTH,
                    beam_size=1,
                )
                pred_texts = tokenizer.batch_decode(preds, skip_special_tokens=True)
                gt_texts   = batch['chosen_text']
                all_preds.extend(pred_texts)
                all_gts.extend(gt_texts)

        val_reward_acc = sum(val_reward_accs) / len(val_reward_accs)
        print(f"📊 Val Reward Accuracy: {val_reward_acc:.3f}")

        metrics = run_full_evaluation(
            all_preds, all_gts,
            model_label=f"DPO Epoch {epoch+1}"
        )
        metrics['reward_acc'] = val_reward_acc

        history.append({
            'epoch':      epoch + 1,
            'train_loss': avg['dpo_loss'],
            'train_racc': avg['reward_acc'],
            **metrics
        })

        # ── Checkpoint ──────────────────────────────────────────────────
        improved = val_reward_acc > best_reward_acc
        if improved:
            best_reward_acc = val_reward_acc
            ckpt = os.path.join(config.CHECKPOINT_DIR, "best_model_dpo.pt")
            torch.save(policy_model.state_dict(), ckpt)
            print(f"✅ DPO Checkpoint saved (reward_acc={val_reward_acc:.3f})")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.PATIENCE:
                print("Early stopping.")
                break

    # ── Lưu history ─────────────────────────────────────────────────────
    import json
    with open("dpo_training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    print(f"\n✅ DPO training done. Best Reward Acc: {best_reward_acc:.3f}")
    return history


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sft_checkpoint",
        default="checkpoints/best_model_A_transformer.pt",
        help="Path to SFT checkpoint (làm reference model)"
    )
    args = parser.parse_args()
    train_dpo(args.sft_checkpoint)
