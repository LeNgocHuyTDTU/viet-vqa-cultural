import json
import os

def check_and_filter_cuisine(input_folder, output_folder):
    files = ['train_data.json', 'val_data.json', 'test_data.json']
    all_data = {}
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    # Bước 1: Lọc dữ liệu ẩm thực
    for file_name in files:
        input_path = os.path.join(input_folder, file_name)
        if not os.path.exists(input_path):
            print(f"⚠️ Không tìm thấy file: {input_path}")
            continue
            
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Chỉ lấy ẩm thực
        filtered = [item for item in data if item.get('category') == 'am_thuc']
        all_data[file_name] = filtered
        
        # Lưu file mới
        output_path = os.path.join(output_folder, file_name)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(filtered, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Đã lọc {file_name}: {len(filtered)} mẫu ẩm thực.")

    print("\n" + "="*50)
    print("🔍 KIỂM TRA TRÙNG LẶP DỮ LIỆU (DATA LEAKAGE)")
    print("="*50)

    # Bước 2: Tạo tập hợp các Key để so sánh
    # Key ở đây là (Đường dẫn ảnh + Câu hỏi)
    splits = {}
    for name, items in all_data.items():
        keys = set()
        for item in items:
            img = item.get('image_path')
            for q_obj in item.get('questions', []):
                q_text = q_obj.get('question')
                # Tạo một dấu vân tay duy nhất cho mỗi cặp Ảnh-Câu hỏi
                keys.add((img, q_text))
        splits[name] = keys

    # Bước 3: So sánh chéo
    pairs = [
        ('train_data.json', 'val_data.json'),
        ('train_data.json', 'test_data.json'),
        ('val_data.json', 'test_data.json')
    ]

    for s1, s2 in pairs:
        if s1 in splits and s2 in splits:
            overlap = splits[s1].intersection(splits[s2])
            count = len(overlap)
            print(f"👉 So sánh [{s1}] và [{s2}]:")
            if count > 0:
                print(f"   ❌ CẢNH BÁO: Phát hiện {count} cặp (Ảnh, Câu hỏi) bị TRÙNG LẶP!")
                # In thử 1 cái bị trùng để đối chứng
                sample = list(overlap)[0]
                print(f"   Ví dụ trùng: Ảnh: {sample[0]} | Câu hỏi: {sample[1]}")
            else:
                print(f"   ✅ Tuyệt vời: Không có sự trùng lặp nào.")
            print("-" * 30)

if __name__ == "__main__":
    input_dir = "Data/splits"
    output_dir = "Data/splits_cuisine" 
    
    check_and_filter_cuisine(input_dir, output_dir)