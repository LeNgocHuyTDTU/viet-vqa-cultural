from config import Config
from dataset import VietVQADataset

config = Config()
train_ds = VietVQADataset(config.TRAIN_JSON, config)

print(f"Tổng số lượng câu hỏi trong tập Train (sau khi lọc): {len(train_ds)}")

# In thử 1 mẫu để kiểm tra xem có đúng là ẩm thực không
if len(train_ds) > 0:
    print("\n--- Mẫu đầu tiên ---")
    sample = train_ds.samples[0]
    print(f"Ảnh: {sample['image_path']}")
    print(f"Câu hỏi: {sample['question']}")
    print(f"Đáp án: {sample['answer']}")
else:
    print("Không tìm thấy dữ liệu nào!")