"""
build_preference_data.py
========================
Tạo preference_data.json từ test_data.json thực tế.
Chạy một lần: python build_preference_data.py

Output: preference_data.json  (2166+ cặp chosen/rejected)
"""

import json, random
from pathlib import Path

random.seed(42)

ALL_FOODS = [
    "banh bao", "banh bot lọc", "banh chay", "banh chung Tet", "banh chung gù",
    "banh cuốn", "banh căn Phan Thiet", "banh day", "banh flan", "banh gai",
    "banh gio", "banh in", "banh it", "banh khọt", "banh mi Việt Nam",
    "banh pia", "banh ran", "banh trang nuớng", "banh tét", "banh xèo",
    "banh đậu xanh", "bún bò Hue", "bún chả Ha Noi", "bún thịt nuớng",
    "ca phê phin", "canh chua", "cao lầu Hoi An", "chao ca", "chao lòng",
    "chè Việt Nam", "chè buởi", "chả ca La Vọng", "chả lụa",
    "chả que Hung Yên", "cơm tấm Sai Gòn", "ga nuớng", "giò thu",
    "goi cuốn", "kẹo dua Ben Tre", "mam ruốc", "mam tôm", "mi Quảng",
    "mut Tet", "nem chua", "nem nuớng", "nuớc mam Phú Quốc",
    "phở Việt Nam", "tôm khô", "vịt quay", "xôi gấc",
]

REJECTED_POOL = {
    "identification": [
        lambda kw: f"Đây là {_other_food(kw)}.",
        lambda kw: "Đây là một món ăn Việt Nam.",
        lambda kw: "Không thể xác định được món ăn này.",
        lambda kw: f"Có thể là {_other_food(kw)} hoặc {_other_food(kw)}.",
        lambda kw: "Đây là món ăn truyền thống.",
        lambda kw: "This is a Vietnamese dish.",
        lambda kw: "Một món ăn ngon.",
        lambda kw: f"Tôi nghĩ đây là {_other_food(kw)}.",
    ],
    "description": [
        lambda kw: "Chỉ có cơm và thịt.",
        lambda kw: f"Gồm {_other_food(kw)} và một số rau.",
        lambda kw: "Nhiều loại thức ăn khác nhau.",
        lambda kw: "Some ingredients and rice.",
        lambda kw: "Không thể mô tả chi tiết.",
        lambda kw: "Có nhiều màu sắc và nguyên liệu.",
        lambda kw: "Gồm rau, thịt, cơm, và các loại gia vị.",
        lambda kw: "Nhiều nguyên liệu được sắp xếp.",
    ],
    "cultural": [
        lambda kw: "Đây là một món ăn ngon của Việt Nam.",
        lambda kw: "Ẩm thực Việt Nam rất đa dạng.",
        lambda kw: "Không có ý nghĩa văn hóa đặc biệt.",
        lambda kw: "Phản ánh văn hóa ẩm thực Hà Nội.",
        lambda kw: "Thể hiện ảnh hưởng của ẩm thực Trung Quốc.",
        lambda kw: "It reflects Vietnamese culinary traditions.",
        lambda kw: "Món ăn hiện đại không có ý nghĩa cổ truyền.",
        lambda kw: "Ý nghĩa văn hóa chưa được xác định rõ.",
    ],
    "analysis": [
        lambda kw: "Vì nó rất ngon và dễ ăn.",
        lambda kw: "Vì nhiều người thích món này.",
        lambda kw: "Không quan trọng lắm.",
        lambda kw: "Because it is a famous dish.",
        lambda kw: "Vì nó rẻ và dễ chế biến.",
        lambda kw: "Vì đây là món ăn của người Hoa.",
        lambda kw: "Chưa có nghiên cứu cụ thể về điều này.",
        lambda kw: "Vì nó phổ biến ở các thành phố lớn.",
    ],
    "comparison": [
        lambda kw: f"Giống hoàn toàn với {_other_food(kw)}.",
        lambda kw: "Không có gì khác biệt.",
        lambda kw: f"Kém hơn {_other_food(kw)} về hương vị.",
        lambda kw: "Similar to dishes in other regions.",
        lambda kw: "Đơn giản hơn các món miền Nam.",
        lambda kw: f"Là biến thể của {_other_food(kw)}.",
        lambda kw: "Khó so sánh vì không có dữ liệu.",
        lambda kw: "Tương tự như các món ăn nước ngoài.",
    ],
}


def _other_food(exclude: str) -> str:
    pool = [f for f in ALL_FOODS if f != exclude]
    return random.choice(pool) if pool else ALL_FOODS[0]


def make_rejected(q_type: str, correct_answer: str, keyword: str) -> str:
    pool = REJECTED_POOL.get(q_type, REJECTED_POOL["identification"])
    fn   = random.choice(pool)
    rej  = fn(keyword)
    # Đảm bảo không trùng chosen
    for _ in range(5):
        if rej.lower().strip() != correct_answer.lower().strip():
            break
        fn  = random.choice(pool)
        rej = fn(keyword)
    return rej


def build(test_json_path: str, out_path: str) -> int:
    with open(test_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = [data]

    pairs = []
    pid   = 0
    for item in data:
        img_path = item.get('image_path', '')
        keyword  = item.get('keyword', '')
        category = item.get('category', '')

        for q in item.get('questions', []):
            q_text    = q.get('question', '').strip()
            q_type    = q.get('question_type', 'identification')
            chosen    = q.get('answer', '').strip()
            difficulty = q.get('difficulty', 'medium')

            if not q_text or not chosen:
                continue

            rejected = make_rejected(q_type, chosen, keyword)

            pairs.append({
                "pair_id":       pid,
                "image_path":    img_path,
                "keyword":       keyword,
                "category":      category,
                "question":      q_text,
                "question_type": q_type,
                "difficulty":    difficulty,
                "chosen":        chosen,         # ✅ ground truth answer
                "rejected":      rejected,        # ❌ bad answer
                "chosen_score":  1.0,
                "rejected_score": 0.0,
            })
            pid += 1

    random.shuffle(pairs)

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(pairs, f, ensure_ascii=False, indent=2)

    print(f"✅ preference_data.json: {len(pairs)} cặp → {out_path}")
    return len(pairs)


if __name__ == "__main__":
    build("./Data/splits_cuisine/test_data.json", "preference_data.json")
