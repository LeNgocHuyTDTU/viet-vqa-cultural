"""
train_ppo.py
============
Proximal Policy Optimization (PPO) cho VQA_Model_A.

Reward = VQARewardModel.compute_batch(pred, gt)
       = α·ExactMatch + β·ROUGE-L + γ·BERTScore − δ·LengthPenalty

Architecture PPO cho seq2seq:
  ┌────────────────────────────────────────────────────────┐
  │  Actor  = VQA_Model_A (policy)  ← cần update          │
  │  Critic = ValueHead(encoder_out) ← ước tính V(s)      │
  │  Reward = VQARewardModel                               │
  └────────────────────────────────────────────────────────┘

Simplified PPO (REINFORCE + KL clip):
  Vì VQA_Model_A là seq2seq nhỏ (không phải LLM), ta dùng
  REINFORCE với baseline (mean reward) + KL divergence penalty
  thay vì full GAE-PPO để tiết kiệm VRAM 4GB.

  L = -E[ (R - b) · log π(a|s) ] + λ_kl · KL(π || π_ref)

  Trong đó:
    R   = reward từ VQARewardModel
    b   = baseline = mean(R) trong batch (giảm variance)
    π   = policy model
    π_ref = reference SFT model (frozen)
    λ_kl = KL penalty coefficient (0.05)

Chạy:
  python train_ppo.py
  python train_ppo.py --sft_checkpoint checkpoints/best_model_A_transformer.pt
"""

import os
import copy
import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from config import Config
from models_A import VQA_Model_A
from dataset_rl import PPOVQADataset
from reward_model import VQARewardModel
from evaluation import run_full_evaluation


# ─────────────────────────────────────────────────────────────────────────────
def compute_token_logprobs(
    model:      VQA_Model_A,
    images:     torch.Tensor,
    q_ids:      torch.Tensor,
    q_mask:     torch.Tensor,
    gen_ids:    list[list[int]],
    bos_id:     int,
    pad_id:     int,
    max_len:    int,
    device:     torch.device,
) -> torch.Tensor:
    """
    Tính log-prob tổng của các token đã sinh ra (gen_ids) dưới model.
    gen_ids : list of token lists (không đồng đều độ dài)
    Returns : (B,) tensor
    """
    B = images.size(0)
    # Pad gen_ids về max_len
    padded = torch.full((B, max_len), pad_id, dtype=torch.long, device=device)
    for i, toks in enumerate(gen_ids):
        t = min(len(toks), max_len)
        padded[i, :t] = torch.tensor(toks[:t], dtype=torch.long, device=device)

    # Prepend BOS
    bos_col = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
    tgt     = torch.cat([bos_col, padded], dim=1)[:, :max_len]  # (B, max_len)

    logits  = model(images, q_ids, q_mask, tgt_ids=tgt)          # (B,T,V)
    log_p   = F.log_softmax(logits[:, :-1, :], dim=-1)           # (B,T-1,V)
    labels  = tgt[:, 1:].clamp(min=0)                            # (B,T-1)

    token_lp = log_p.gather(2, labels.unsqueeze(2)).squeeze(2)   # (B,T-1)
    mask     = (tgt[:, 1:] != pad_id).float()
    return (token_lp * mask).sum(dim=1)                          # (B,)


# ─────────────────────────────────────────────────────────────────────────────
def train_ppo(sft_checkpoint: str | None = None):
    config = Config()

    # ── Dataset ─────────────────────────────────────────────────────────────
    full_ds = PPOVQADataset("preference_data.json", config)
    val_size   = max(1, int(0.1 * len(full_ds)))
    train_size = len(full_ds) - val_size
    train_ds, val_ds = random_split(full_ds, [train_size, val_size])

    # PPO thường dùng batch nhỏ hơn (rollout cost cao)
    train_loader = DataLoader(train_ds, batch_size=4,  shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=8,  shuffle=False,
                              num_workers=2, pin_memory=True)

    tokenizer  = full_ds.tokenizer
    vocab_size = tokenizer.vocab_size
    pad_id     = tokenizer.pad_token_id or 1
    bos_id     = tokenizer.cls_token_id or 0
    eos_id     = tokenizer.sep_token_id or 2

    # ── Reference Model (SFT, frozen) ───────────────────────────────────────
    ref_model = VQA_Model_A(config, vocab_size).to(config.DEVICE)
    if sft_checkpoint and os.path.exists(sft_checkpoint):
        ref_model.load_state_dict(
            torch.load(sft_checkpoint, map_location=config.DEVICE)
        )
        print(f"✅ Reference model loaded: {sft_checkpoint}")
    else:
        print("⚠️  SFT checkpoint không tìm thấy — dùng random init.")

    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    # ── Policy Model (trainable) ─────────────────────────────────────────────
    policy_model = copy.deepcopy(ref_model).to(config.DEVICE)
    for name, p in policy_model.named_parameters():
        if any(k in name for k in ['transformer_decoder', 'lstm', 'fc_out',
                                    'cross_attn', 'token_embedding',
                                    'img_spatial_proj', 'img_pool_proj', 'txt_proj']):
            p.requires_grad = True
        else:
            p.requires_grad = False

    # ── Reward Model ─────────────────────────────────────────────────────────
    reward_model = VQARewardModel(alpha=0.4, beta=0.3, gamma=0.3, delta=0.05)

    optimizer = torch.optim.AdamW(
        [p for p in policy_model.parameters() if p.requires_grad],
        lr=3e-5, weight_decay=1e-2
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.EPOCHS
    )
    scaler = torch.amp.GradScaler('cuda')

    # Hyperparams PPO
    lambda_kl  = 0.05    # KL penalty coefficient
    clip_ratio = 0.2     # PPO clip ratio (unused in REINFORCE variant, kept for reference)

    best_vqa_acc = 0.0
    patience_counter = 0
    history = []
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  PPO Training (REINFORCE + KL)  |  λ_kl={lambda_kl}")
    print(f"  {train_size} train  |  {val_size} val")
    print(f"{'='*60}\n")

    for epoch in range(config.EPOCHS):

        # ── Rollout + Update ─────────────────────────────────────────────
        policy_model.eval()   # generate() requires eval mode
        all_rewards = []
        batch_data  = []      # buffer rollouts

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [PPO Rollout]")
        for batch in pbar:
            imgs   = batch['image'].to(config.DEVICE)
            q_ids  = batch['question_ids'].to(config.DEVICE)
            q_mask = batch['question_mask'].to(config.DEVICE)
            gt_texts = batch['answer_text']

            # ── 1. Rollout: generate với current policy ──────────────
            with torch.no_grad():
                gen_ids = policy_model.generate(
                    imgs, q_ids, q_mask,
                    bos_token_id=bos_id,
                    eos_token_id=eos_id,
                    max_len=config.MAX_ANSWER_LENGTH,
                    beam_size=1,
                )
            pred_texts = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)

            # ── 2. Reward ────────────────────────────────────────────
            rewards = reward_model.compute_batch(pred_texts, gt_texts)
            all_rewards.extend(rewards)

            batch_data.append({
                'imgs': imgs.cpu(), 'q_ids': q_ids.cpu(), 'q_mask': q_mask.cpu(),
                'gen_ids': gen_ids, 'rewards': rewards,
            })

            pbar.set_postfix(avg_r=f"{sum(rewards)/len(rewards):.3f}")

        # Baseline = mean reward cả epoch
        baseline = sum(all_rewards) / len(all_rewards)

        # ── 3. Policy Update ─────────────────────────────────────────
        policy_model.train()
        epoch_losses = []

        for bd in batch_data:
            imgs   = bd['imgs'].to(config.DEVICE)
            q_ids  = bd['q_ids'].to(config.DEVICE)
            q_mask = bd['q_mask'].to(config.DEVICE)
            rewards_t = torch.tensor(bd['rewards'], dtype=torch.float32,
                                     device=config.DEVICE)
            advantages = rewards_t - baseline           # (B,)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda'):
                # Policy log-probs
                pi_logp = compute_token_logprobs(
                    policy_model, imgs, q_ids, q_mask,
                    bd['gen_ids'], bos_id, pad_id,
                    config.MAX_ANSWER_LENGTH, config.DEVICE
                )                                       # (B,)

            # Reference log-probs (no grad)
            with torch.no_grad():
                ref_logp = compute_token_logprobs(
                    ref_model, imgs, q_ids, q_mask,
                    bd['gen_ids'], bos_id, pad_id,
                    config.MAX_ANSWER_LENGTH, config.DEVICE
                )

            with torch.amp.autocast('cuda'):
                # REINFORCE loss
                reinforce_loss = -(advantages.detach() * pi_logp).mean()

                # KL penalty: KL(policy || ref) ≈ log(policy) - log(ref)
                kl_penalty = (pi_logp - ref_logp).mean()
                kl_penalty = torch.clamp(kl_penalty, min=0)  # chỉ phạt khi drift xa

                loss = reinforce_loss + lambda_kl * kl_penalty

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_losses.append(loss.item())

        scheduler.step()

        rw_stats = reward_model.summary_stats(all_rewards)
        print(f"\n📈 Epoch {epoch+1} PPO — "
              f"Loss: {sum(epoch_losses)/len(epoch_losses):.4f} | "
              f"Reward mean: {rw_stats['mean']:.4f} ± {rw_stats['std']:.4f} | "
              f"Pos rate: {rw_stats['pos_rate']:.3f} | "
              f"Baseline: {baseline:.4f}")

        # ── Validation ───────────────────────────────────────────────
        policy_model.eval()
        all_preds, all_gts = [], []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1} [PPO Val]"):
                imgs   = batch['image'].to(config.DEVICE)
                q_ids  = batch['question_ids'].to(config.DEVICE)
                q_mask = batch['question_mask'].to(config.DEVICE)

                gen_ids = policy_model.generate(
                    imgs, q_ids, q_mask,
                    bos_token_id=bos_id,
                    eos_token_id=eos_id,
                    max_len=config.MAX_ANSWER_LENGTH,
                    beam_size=1,
                )
                pred_texts = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
                all_preds.extend(pred_texts)
                all_gts.extend(batch['answer_text'])

        metrics = run_full_evaluation(
            all_preds, all_gts,
            model_label=f"PPO Epoch {epoch+1}"
        )
        metrics.update({
            'reward_mean': rw_stats['mean'],
            'reward_std':  rw_stats['std'],
            'pos_rate':    rw_stats['pos_rate'],
        })

        history.append({'epoch': epoch+1, **metrics})

        # ── Checkpoint ──────────────────────────────────────────────
        if metrics['vqa_acc'] > best_vqa_acc:
            best_vqa_acc = metrics['vqa_acc']
            ckpt = os.path.join(config.CHECKPOINT_DIR, "best_model_ppo.pt")
            torch.save(policy_model.state_dict(), ckpt)
            print(f"✅ PPO Checkpoint saved (VQA Acc={best_vqa_acc:.2f}%)")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.PATIENCE:
                print("Early stopping.")
                break

    with open("ppo_training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    print(f"\n✅ PPO done. Best VQA Acc: {best_vqa_acc:.2f}%")
    return history


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sft_checkpoint",
        default="checkpoints/best_model_A_transformer.pt",
    )
    args = parser.parse_args()
    train_ppo(args.sft_checkpoint)
