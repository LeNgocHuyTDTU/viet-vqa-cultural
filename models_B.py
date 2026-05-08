import torch
from config import Config
from transformers import ViltProcessor, ViltForQuestionAnswering
from peft import LoraConfig, get_peft_model
from PIL import Image

class VLM_Pipeline:
    """
    Hướng B: ViLT-based VQA Pipeline (Classification).
    """
    def __init__(self, id2label: dict, label2id: dict, mode: str = "zero_shot"):
        self.config   = Config()
        self.id2label = id2label
        self.label2id = label2id

        self.processor = ViltProcessor.from_pretrained(self.config.VLM_MODEL_ID)

        self.model = ViltForQuestionAnswering.from_pretrained(
            self.config.VLM_MODEL_ID,
            num_labels=len(id2label),
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True, 
        ).to(self.config.DEVICE)

        if mode == "fine_tune":
            # B2: LoRA Fine-tuning — target_modules đúng với ViLT attention
            lora_config = LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=["query", "value"], 
                lora_dropout=0.05,
                bias="none"
            )
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
            print("✅ Đã tải cấu hình B2 (LoRA Fine-tuned).")
        else:
            # Freeze toàn bộ model cho zero-shot
            for param in self.model.parameters():
                param.requires_grad = False
            print("✅ Đã tải cấu hình B1 (Zero-shot).")

    def generate_answer(self, image_path: str, question: str) -> str:
        """
        ViLT là classifier — Trả về nhãn có xác suất cao nhất từ id2label.
        """
        image = Image.open(image_path).convert("RGB")

        inputs = self.processor(
            images=image,
            text=question,
            return_tensors="pt",
            truncation=True
        ).to(self.config.DEVICE)

        with torch.no_grad():
            outputs = self.model(**inputs)

        pred_id  = outputs.logits.argmax(-1).item()
        answer   = self.id2label.get(pred_id, "không rõ")
        return answer

    def batch_predict(self, image_paths: list, questions: list) -> list:
        answers = []
        for img_path, question in zip(image_paths, questions):
            try:
                ans = self.generate_answer(img_path, question)
            except Exception as e:
                print(f"⚠️ Lỗi khi xử lý {img_path}: {e}")
                ans = "lỗi"
            answers.append(ans)
        return answers