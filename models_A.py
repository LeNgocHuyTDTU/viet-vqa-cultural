import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from transformers import AutoModel

# ── [107-108] FlashAttention via PyTorch 2.x ──────────────────────────────────
torch.backends.cuda.enable_flash_sdp(True)


class CrossAttentionFusion(nn.Module):
    """
    [114] Memory Augmentation — Cross-Attention vào spatial image patches.
    Decoder attend vào 49 patch features (7x7) của ResNet layer4 thay vì
    chỉ dùng 1 pooled vector → model "nhìn" được chi tiết vùng ảnh liên quan.
    """
    def __init__(self, hidden_dim: int, n_heads: int = 8):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=n_heads, dropout=0.1, batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, query: torch.Tensor, image_memory: torch.Tensor) -> torch.Tensor:
        attended, _ = self.attn(query, image_memory, image_memory)
        return self.norm(query + attended)


class VQA_Model_A(nn.Module):
    """
    Hướng A — ResNet50 + PhoBERT + LSTM/Transformer Decoder.

    Kỹ thuật áp dụng:
      [107-108] FlashAttention  : enable_flash_sdp(True)
      [114] Memory Augmentation : CrossAttentionFusion → spatial image patches
      [120] Prefix LM           : Freeze encoder, chỉ train decoder + projections
      [122] Denoising           : Label Smoothing trong CrossEntropyLoss (train_A.py)
      [140] Constrained Decoding: Beam Search trong generate()
    """

    def __init__(self, config, vocab_size: int):
        super().__init__()
        self.config       = config
        self.decoder_type = config.DECODER_TYPE
        self.vocab_size   = vocab_size
        self.hidden_dim   = config.HIDDEN_DIM

        # ── Image Encoder: lấy spatial map 7×7 ─────────────────────────────
        resnet = models.resnet50(weights='DEFAULT')
        self.image_encoder_spatial = nn.Sequential(*list(resnet.children())[:-2])  # (B,2048,7,7)
        self.img_spatial_proj      = nn.Linear(2048, self.hidden_dim)
        self.img_pool_proj         = nn.Linear(2048, self.hidden_dim)

        # ── Text Encoder (PhoBERT) ──────────────────────────────────────────
        self.text_encoder = AutoModel.from_pretrained(config.TEXT_ENCODER)
        self.txt_proj     = nn.Linear(768, self.hidden_dim)

        # [120] Prefix LM: Freeze encoder — chỉ train decoder và projections
        for p in self.text_encoder.parameters():
            p.requires_grad = False
        for p in self.image_encoder_spatial.parameters():
            p.requires_grad = False

        # ── Token Embedding ─────────────────────────────────────────────────
        self.token_embedding = nn.Embedding(vocab_size, self.hidden_dim)

        # ── [114] Cross-Attention Fusion ─────────────────────────────────────
        self.cross_attn = CrossAttentionFusion(self.hidden_dim, n_heads=8)

        # ── Decoder ─────────────────────────────────────────────────────────
        if self.decoder_type == 'lstm':
            self.lstm = nn.LSTM(
                input_size=self.hidden_dim, hidden_size=self.hidden_dim,
                num_layers=2, batch_first=True, dropout=0.3
            )
            self.context_to_h = nn.Linear(self.hidden_dim * 2, self.hidden_dim * 2)
            self.context_to_c = nn.Linear(self.hidden_dim * 2, self.hidden_dim * 2)
        else:
            dec_layer = nn.TransformerDecoderLayer(
                d_model=self.hidden_dim, nhead=8,
                dim_feedforward=512, dropout=0.1, batch_first=True
            )
            self.transformer_decoder = nn.TransformerDecoder(dec_layer, num_layers=4)

        self.fc_out  = nn.Linear(self.hidden_dim, vocab_size)
        self.dropout = nn.Dropout(0.1)

    # ── Encode ──────────────────────────────────────────────────────────────
    def encode(self, images, q_ids, q_mask):
        with torch.no_grad():
            spatial = self.image_encoder_spatial(images)           # (B,2048,7,7)
        patches      = spatial.flatten(2).permute(0, 2, 1)         # (B,49,2048)
        img_patches  = self.img_spatial_proj(patches)              # (B,49,D)
        img_global   = self.img_pool_proj(patches.mean(1)).unsqueeze(1)  # (B,1,D)

        with torch.no_grad():
            txt_raw  = self.text_encoder(q_ids, attention_mask=q_mask).last_hidden_state
        txt_feat = self.txt_proj(txt_raw)                          # (B,L,D)
        return img_patches, img_global, txt_feat

    # ── Forward (Teacher Forcing) ────────────────────────────────────────────
    def forward(self, images, q_ids, q_mask, tgt_ids):
        img_patches, img_global, txt_feat = self.encode(images, q_ids, q_mask)
        memory  = torch.cat([img_global, txt_feat], dim=1)         # (B,1+L,D)
        tgt_emb = self.dropout(self.token_embedding(tgt_ids))      # (B,T,D)

        if self.decoder_type == 'lstm':
            ctx = torch.cat([img_global.squeeze(1), txt_feat[:, 0, :]], dim=1)
            h0  = self.context_to_h(ctx).view(-1,2,self.hidden_dim).permute(1,0,2).contiguous()
            c0  = self.context_to_c(ctx).view(-1,2,self.hidden_dim).permute(1,0,2).contiguous()
            out, _ = self.lstm(tgt_emb, (h0, c0))
        else:
            T    = tgt_emb.size(1)
            mask = (nn.Transformer.generate_square_subsequent_mask(T, device=tgt_emb.device) == float('-inf'))

            # 1. Tạo mask cho target (bỏ qua các token <pad>)
            tgt_pad_mask = (tgt_ids == 1) # 1 là pad_token_id của PhoBERT

            # 2. Tạo mask cho memory (câu hỏi + ảnh). q_mask có giá trị 1 (real), 0 (pad)
            # Ta cần chuyển 0 -> True, 1 -> False theo chuẩn PyTorch
            text_pad_mask = (q_mask == 0) 
            img_pad_mask = torch.zeros(img_global.size(0), 1, dtype=torch.bool, device=q_mask.device)
            mem_pad_mask = torch.cat([img_pad_mask, text_pad_mask], dim=1)

            out  = self.transformer_decoder(
                tgt_emb, 
                memory, 
                tgt_mask=mask, 
                tgt_key_padding_mask=tgt_pad_mask,
                memory_key_padding_mask=mem_pad_mask
                )

        out    = self.cross_attn(out, img_patches)  # [114] attend vào spatial patches
        return self.fc_out(out)                     # (B,T,vocab)

    # ── Generate ─────────────────────────────────────────────────────────────
    @torch.no_grad()
    def generate(self, images, q_ids, q_mask, bos_token_id, eos_token_id,
                 max_len=30, beam_size=4):
        """[140] Beam Search (beam_size=1 → greedy)."""
        self.eval()
        B = images.size(0)
        img_patches, img_global, txt_feat = self.encode(images, q_ids, q_mask)

        if beam_size == 1:
            return self._greedy(B, img_patches, img_global, txt_feat,
                                bos_token_id, eos_token_id, max_len)
        results = []
        for i in range(B):
            best = self._beam_single(
                img_patches[i:i+1], img_global[i:i+1], txt_feat[i:i+1],
                bos_token_id, eos_token_id, max_len, beam_size
            )
            results.append(best)
        return results

    def _greedy(self, B, img_patches, img_global, txt_feat, bos_id, eos_id, max_len):
        dev      = img_global.device
        memory   = torch.cat([img_global, txt_feat], dim=1)
        gen      = torch.full((B, 1), bos_id, dtype=torch.long, device=dev)
        finished = torch.zeros(B, dtype=torch.bool, device=dev)

        if self.decoder_type == 'lstm':
            ctx = torch.cat([img_global.squeeze(1), txt_feat[:, 0, :]], dim=1)
            h   = self.context_to_h(ctx).view(-1,2,self.hidden_dim).permute(1,0,2).contiguous()
            c   = self.context_to_c(ctx).view(-1,2,self.hidden_dim).permute(1,0,2).contiguous()
            for _ in range(max_len):
                tgt       = self.token_embedding(gen[:, -1:])
                out, (h,c)= self.lstm(tgt, (h, c))
                out       = self.cross_attn(out, img_patches)
                tok       = self.fc_out(out).argmax(-1)
                finished |= tok.squeeze(1) == eos_id
                gen       = torch.cat([gen, tok], dim=1)
                if finished.all(): break
        else:
            for _ in range(max_len):
                T    = gen.size(1)
                tgt  = self.token_embedding(gen)
                mask = nn.Transformer.generate_square_subsequent_mask(T, device=dev)
                out  = self.transformer_decoder(tgt, memory, tgt_mask=mask)
                out  = self.cross_attn(out, img_patches)
                tok  = self.fc_out(out[:, -1:, :]).argmax(-1)
                finished |= tok.squeeze(1) == eos_id
                gen  = torch.cat([gen, tok], dim=1)
                if finished.all(): break

        return gen[:, 1:].tolist()

    def _beam_single(self, img_patches, img_global, txt_feat,
                     bos_id, eos_id, max_len, beam_size):
        dev     = img_global.device
        memory  = torch.cat([img_global, txt_feat], dim=1)
        beams   = [(0.0, [bos_id])]
        done    = []

        for _ in range(max_len):
            cands = []
            for lp, toks in beams:
                if toks[-1] == eos_id:
                    done.append((lp, toks)); continue

                seq = torch.tensor([toks], dtype=torch.long, device=dev)
                emb = self.token_embedding(seq)

                if self.decoder_type == 'lstm':
                    ctx = torch.cat([img_global.squeeze(1), txt_feat[:, 0, :]], dim=1)
                    h      = self.context_to_h(ctx).view(-1,2,self.hidden_dim).permute(1,0,2).contiguous()
                    c      = self.context_to_c(ctx).view(-1,2,self.hidden_dim).permute(1,0,2).contiguous()
                    out, _ = self.lstm(emb, (h, c))
                else:
                    T    = seq.size(1)
                    mask = nn.Transformer.generate_square_subsequent_mask(T, device=dev)
                    out  = self.transformer_decoder(emb, memory, tgt_mask=mask)

                out  = self.cross_attn(out, img_patches)
                lps  = F.log_softmax(self.fc_out(out[0, -1]), dim=-1)
                top_lp, top_id = lps.topk(beam_size)
                for tlp, tid in zip(top_lp.tolist(), top_id.tolist()):
                    cands.append((lp + tlp, toks + [tid]))

            if not cands: break
            cands.sort(key=lambda x: x[0] / max(len(x[1]), 1), reverse=True)
            beams = cands[:beam_size]

        all_hyps = done + beams
        all_hyps.sort(key=lambda x: x[0] / max(len(x[1]), 1), reverse=True)
        best = all_hyps[0][1][1:]  # bỏ BOS
        if eos_id in best:
            best = best[:best.index(eos_id)]
        return best