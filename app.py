import os
import json
import torch
import warnings
import logging
import re
from typing import Any
import streamlit as st
from PIL import Image
try:
    from transformers import AutoProcessor, AutoModelForVisualQuestionAnswering
except ImportError:
    # Fallback cho transformers bản cũ chưa có AutoModelForVisualQuestionAnswering.
    from transformers import ViltProcessor as AutoProcessor  # type: ignore
    from transformers import ViltForQuestionAnswering as AutoModelForVisualQuestionAnswering  # type: ignore
from peft import PeftModel
from config import Config
from models_A import VQA_Model_A
from dataset import VietVQADataset

# --- CHẶN TOÀN BỘ CẢNH BÁO RÁC ---
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("transformers").setLevel(logging.ERROR) # Khóa Hugging Face warning

# --- CẤU HÌNH GIAO DIỆN WEB ---
st.set_page_config(page_title="Việt VQA Ẩm Thực", layout="wide")
st.title("Việt VQA: So Sánh 4 Kiến Trúc AI")
st.markdown("**Đồ án Deep Learning** - Trực tiếp so sánh hiệu năng giữa LSTM, Transformer và ViLT.")
st.markdown("---")

# --- LÕI XỬ LÝ ---
class VQA_Arena:
    def __init__(self):
        self.cfg = Config()
        self.device = self.cfg.DEVICE
        # Lấy thư mục hiện tại làm gốc để tránh lỗi đường dẫn tuyệt đối/tương đối
        self.ckpt_dir = self.cfg.CHECKPOINT_DIR.rstrip('/') 
        
        # 1. Hướng A
        self.temp_ds = VietVQADataset(self.cfg.VAL_JSON, self.cfg)
        self.vocab_size = self.temp_ds.tokenizer.vocab_size
        
        # 2. Hướng B
        self.b_processor = AutoProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
        self.id2label, self.label2id = self.build_vocab_b()
        self.qa_by_image, self.dish_label_ids = self.build_reference_knowledge()
        
        # Lưu các mô hình đã nạp (lazy loading)
        self.models = {}

        # Tham số decoding mặc định (có thể override từ `Config`)
        self.decoding_strategy = getattr(self.cfg, 'DECODING_STRATEGY', {
            "max_new_tokens":    15,
            "do_sample":         True,
            "temperature":       0.2,
            "top_k":             40,
            "top_p":             0.9,
            "repetition_penalty": 1.15
        })

        # Không tự động load tất cả mô hình nữa — sẽ load khi cần
        # self.load_all_models()

    def _clean_answer(self, text):
        text = str(text).replace("</s>", " ").replace("<s>", " ").strip()
        text = " ".join(text.split())
        if not text:
            return "không rõ"

        # Gộp bớt các từ lặp liên tiếp để giảm hiện tượng lặp token khi decode.
        words = text.split(" ")
        dedup_words = [words[0]]
        for w in words[1:]:
            if w != dedup_words[-1]:
                dedup_words.append(w)
        return " ".join(dedup_words)

    def _norm_text(self, text):
        text = str(text).lower().strip()
        text = re.sub(r"\s+", " ", text)
        return text

    def _is_dish_name_question(self, question):
        q = self._norm_text(question)
        keywords = [
            "đây là món gì",
            "món gì",
            "tên món",
            "món ăn gì",
            "đây là gì"
        ]
        return any(k in q for k in keywords)

    def build_vocab_b(self):
        unique_answers = set()
        for path in [self.cfg.TRAIN_JSON, self.cfg.VAL_JSON, self.cfg.TEST_JSON]:
            if not os.path.exists(path): continue
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for item in data:
                    if item.get('category') == 'am_thuc':
                        for q in item.get('questions', []):
                            ans = str(q['answer']).lower().strip()
                            unique_answers.add(ans)
        unique_answers = sorted(list(unique_answers))
        return {i: label for i, label in enumerate(unique_answers)}, {label: i for i, label in enumerate(unique_answers)}

    def build_reference_knowledge(self):
        qa_by_image = {}
        dish_label_ids = set()

        for path in [self.cfg.TRAIN_JSON, self.cfg.VAL_JSON, self.cfg.TEST_JSON]:
            if not os.path.exists(path):
                continue

            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            for item in data:
                if item.get('category') != 'am_thuc':
                    continue

                img_path = item.get('image_path', '')
                img_name = os.path.basename(img_path)

                for q in item.get('questions', []):
                    q_text = str(q.get('question', ''))
                    ans_text = str(q.get('answer', '')).strip()
                    ans_key = ans_text.lower().strip()

                    qa_by_image.setdefault(img_name, []).append((self._norm_text(q_text), ans_text))

                    if self._is_dish_name_question(q_text):
                        ans_idx = self.label2id.get(ans_key)
                        if ans_idx is not None:
                            dish_label_ids.add(ans_idx)

        return qa_by_image, dish_label_ids

    def lookup_reference_answer(self, image_name, question):
        if not image_name:
            return None

        candidates = self.qa_by_image.get(os.path.basename(image_name), [])
        if not candidates:
            return None

        norm_q = self._norm_text(question)
        for q_text, ans in candidates:
            if q_text == norm_q:
                return ans

        if self._is_dish_name_question(question):
            for q_text, ans in candidates:
                if self._is_dish_name_question(q_text):
                    return ans

        return None

    def load_all_models(self):
        # Đã chuyển sang lazy-loading; nếu cần load tất cả thì gọi `ensure_model_loaded` từng model.
        return

    def ensure_model_loaded(self, model_name: str):
        """Load a single model on demand if not already loaded."""
        if model_name in self.models:
            return

        try:
            if model_name == 'A1':
                self.cfg.DECODER_TYPE = "lstm"
                a1 = VQA_Model_A(self.cfg, self.vocab_size)
                a1.load_state_dict(torch.load(f"{self.ckpt_dir}/best_model_A1.pt", map_location='cpu'))
                self.models['A1'] = a1

            elif model_name == 'A2':
                self.cfg.DECODER_TYPE = "transformer"
                a2 = VQA_Model_A(self.cfg, self.vocab_size)
                a2.load_state_dict(torch.load(f"{self.ckpt_dir}/best_model_A2.pt", map_location='cpu'))
                self.models['A2'] = a2

            elif model_name == 'B1':
                b1 = AutoModelForVisualQuestionAnswering.from_pretrained(f"{self.ckpt_dir}/best_model_B1")
                self.models['B1'] = b1

            elif model_name == 'B2':
                base_model = AutoModelForVisualQuestionAnswering.from_pretrained(f"{self.ckpt_dir}/best_model_B1")
                b2 = PeftModel.from_pretrained(base_model, f"{self.ckpt_dir}/best_model_B2")
                self.models['B2'] = b2

        except Exception as e:
            print(f"Lỗi tải {model_name}: {e}")

    def predict_A(self, model_name, image, question, decoding_kwargs: dict | None = None):
        model = self.models.get(model_name)
        if model is None: return "⚠️ Chưa có Model"
        
        model.to(self.device)
        model.eval()
        with torch.no_grad():
            img_tensor = self.temp_ds.transform(image).unsqueeze(0).to(self.device)
            max_q_len = getattr(self.cfg, 'MAX_SEQ_LENGTH', getattr(self.cfg, 'MAX_Q_LEN', 30))
            max_ans_len = getattr(self.cfg, 'MAX_ANSWER_LENGTH', 15)
            enc = self.temp_ds.tokenizer(
                question,
                return_tensors='pt',
                padding='max_length',
                max_length=max_q_len,
                truncation=True
            )
            # THAY THẾ BẰNG ĐOẠN NÀY:
            tokenizer = self.temp_ds.tokenizer
            bos_id = tokenizer.cls_token_id or tokenizer.bos_token_id or 0
            eos_id = tokenizer.sep_token_id or tokenizer.eos_token_id or 2

            # Dùng hàm generate thay vì forward
            user_kwargs = decoding_kwargs if decoding_kwargs is not None else {}
            gen_kwargs = {**self.decoding_strategy, **user_kwargs}

            # Map to older/custom model.generate arg names if needed
            gen_for_model = {}
            # prefer max_len (old custom arg) mapped from max_new_tokens
            if 'max_new_tokens' in gen_kwargs:
                gen_for_model['max_len'] = int(gen_kwargs.get('max_new_tokens'))
            elif 'max_len' in gen_kwargs:
                gen_for_model['max_len'] = int(gen_kwargs.get('max_len'))

            # beam size fallback: if sampling is off, use beam search with beam_size=1 by default
            if 'beam_size' in gen_kwargs:
                gen_for_model['beam_size'] = int(gen_kwargs.get('beam_size'))
            else:
                gen_for_model['beam_size'] = 1 if not gen_kwargs.get('do_sample', False) else 1

            # common sampling params (may be ignored by custom generate)
            for k in ['do_sample', 'temperature', 'top_k', 'top_p', 'repetition_penalty']:
                if k in gen_kwargs:
                    gen_for_model[k] = gen_kwargs[k]

            try:
                pred_token_lists = model.generate(
                    images=img_tensor,
                    q_ids=enc['input_ids'].to(self.device),
                    q_mask=enc['attention_mask'].to(self.device),
                    bos_token_id=bos_id,
                    eos_token_id=eos_id,
                    **gen_for_model
                )
            except TypeError:
                # Fallback: call generate with minimal args (older API)
                pred_token_lists = model.generate(
                    images=img_tensor,
                    q_ids=enc['input_ids'].to(self.device),
                    q_mask=enc['attention_mask'].to(self.device),
                    bos_token_id=bos_id,
                    eos_token_id=eos_id,
                    max_len=gen_for_model.get('max_len', 30),
                    beam_size=gen_for_model.get('beam_size', 1)
                )

            decoded = tokenizer.batch_decode(pred_token_lists, skip_special_tokens=True)
            # `batch_decode` returns a list; use the first prediction as the answer string
            if isinstance(decoded, list):
                ans_text = decoded[0] if len(decoded) > 0 else ""
            else:
                ans_text = str(decoded)

            ans_text = self._clean_answer(ans_text)
        model.to('cpu')
        return ans_text

    def predict_B(self, model_name, image, question):
        model = self.models.get(model_name)
        if model is None: return "⚠️ Chưa có Model"
        
        model.to(self.device)
        model.eval()
        with torch.no_grad():
            inputs: Any = self.b_processor(image, question, return_tensors="pt")  # type: ignore[call-arg]
            inputs = inputs.to(self.device)
            outputs = model(**inputs)
            logits = outputs.logits[0]
            predicted_id: int

            if self._is_dish_name_question(question) and self.dish_label_ids:
                constrained_logits = torch.full_like(logits, float("-inf"))
                allowed_ids = list(self.dish_label_ids)
                constrained_logits[allowed_ids] = logits[allowed_ids]
                predicted_id = int(constrained_logits.argmax(-1).item())
            else:
                predicted_id = int(logits.argmax(-1).item())

            ans = self.id2label.get(predicted_id)
            if ans is None:
                model_id2label = getattr(model.config, "id2label", {})
                ans = model_id2label.get(predicted_id) or model_id2label.get(str(predicted_id), "không rõ")
            ans = self._clean_answer(ans)
        model.to('cpu')
        return ans

# --- KHỞI TẠO CACHE ---
@st.cache_resource
def get_arena(cache_key: str = "v3"):
    return VQA_Arena()

with st.spinner("⏳ Đang nạp cả 4 mô hình vào CPU... Chỉ mất 1 lần duy nhất!"):
    arena = get_arena("v3")

# --- GIAO DIỆN TƯƠNG TÁC ---
col_img, col_input = st.columns([1, 1])
image = None

with col_img:
    st.subheader("1. Tải ảnh món ăn")
    uploaded_file = st.file_uploader("Chọn ảnh từ máy...", type=["jpg", "jpeg", "png"])
    uploaded_name = None
    if uploaded_file:
        uploaded_name = uploaded_file.name
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, width='stretch')

with col_input:
    st.subheader("2. Đặt câu hỏi")
    question = st.text_input("Gõ câu hỏi vào đây:", placeholder="Món này nấu bằng nguyên liệu gì?")
    st.markdown("---")

    # Chọn mô hình để chạy (lazy load)
    models_selected = st.multiselect("Chọn mô hình để chạy", options=["A1", "A2", "B1", "B2"], default=["A2"])

    # Tham số decoding cho A1/A2 (UI để tinh chỉnh)
    with st.expander("Tham số decoding (ảnh hưởng A1/A2)"):
        ds = arena.decoding_strategy
        max_new_tokens = st.number_input("max_new_tokens", min_value=1, max_value=256, value=int(ds.get('max_new_tokens', 15)))
        do_sample = st.checkbox("do_sample", value=bool(ds.get('do_sample', True)))
        temperature = st.number_input("temperature", min_value=0.01, max_value=5.0, value=float(ds.get('temperature', 0.2)), format="%.3f")
        top_k = st.number_input("top_k", min_value=0, max_value=1000, value=int(ds.get('top_k', 40)))
        top_p = st.number_input("top_p", min_value=0.0, max_value=1.0, value=float(ds.get('top_p', 0.9)), format="%.3f")
        repetition_penalty = st.number_input("repetition_penalty", min_value=0.1, max_value=5.0, value=float(ds.get('repetition_penalty', 1.15)), format="%.3f")

    if st.button("Chạy mô hình đã chọn", width='stretch'):
        if image is None or not question.strip():
            st.warning("⚠️ Cần phải có cả Ảnh và Câu hỏi!")
        else:
            # Tạo dict tham số để truyền vào predict_A
            decoding_overrides = {
                'max_new_tokens': int(max_new_tokens),
                'do_sample': bool(do_sample),
                'temperature': float(temperature),
                'top_k': int(top_k),
                'top_p': float(top_p),
                'repetition_penalty': float(repetition_penalty)
            }

            results = {"A1": "-", "A2": "-", "B1": "-", "B2": "-"}

            # Thực thi dự đoán cho từng mô hình đã chọn (lazy-load khi cần)
            with st.spinner("🤖 Đang tải mô hình được chọn và dự đoán..."):
                for m in models_selected:
                    arena.ensure_model_loaded(m)
                    if m in ['A1', 'A2']:
                        try:
                            res = arena.predict_A(m, image, question, decoding_overrides)
                        except Exception as e:
                            res = f"Lỗi: {e}"
                    else:
                        try:
                            res = arena.predict_B(m, image, question)
                        except Exception as e:
                            res = f"Lỗi: {e}"
                    results[m] = res

                ref_answer = arena.lookup_reference_answer(uploaded_name, question)

            st.success("✅ Hoàn tất dự đoán!")

            # --- HIỂN THỊ KẾT QUẢ ---
            st.markdown("### Kết quả thực thi:")
            c1, c2, c3, c4 = st.columns(4)

            # A1
            if 'A1' in models_selected:
                c1.info(f"**A1 (LSTM)**\n\n{str(results['A1']).capitalize()}")
            else:
                c1.info("**A1 (LSTM)**\n\n_Không chọn_")

            # A2
            if 'A2' in models_selected:
                c2.success(f"**A2 (Transformer)**\n\n{str(results['A2']).capitalize()}")
            else:
                c2.success("**A2 (Transformer)**\n\n_Không chọn_")

            # B1
            if 'B1' in models_selected:
                c3.warning(f"**B1 (ViLT Gốc)**\n\n{str(results['B1']).capitalize()}")
            else:
                c3.warning("**B1 (ViLT Gốc)**\n\n_Không chọn_")

            # B2
            if 'B2' in models_selected:
                c4.error(f"**B2 (ViLT Fine-tuned)**\n\n{str(results['B2']).capitalize()}")
            else:
                c4.error("**B2 (ViLT Fine-tuned)**\n\n_Không chọn_")