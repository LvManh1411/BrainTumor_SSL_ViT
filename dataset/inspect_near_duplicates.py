"""
inspect_near_duplicates.py

Công cụ REVIEW TRỰC QUAN - hiển thị ảnh cạnh nhau để mắt thường xác nhận
các nhóm được gắn nhãn 'near' (near-duplicate qua pHash) có thực sự là
trùng lặp hay không, TRƯỚC KHI tin tưởng đề xuất xóa.

Lý do cần bước này: ảnh MRI có đặc điểm các lát cắt (slice) liền kề của
cùng 1 ca chụp thường nhìn rất giống nhau, nhưng đây là 2 mẫu dữ liệu
KHÁC NHAU và đều hợp lệ - không phải trùng lặp thật. Nếu ngưỡng pHash
quá lỏng, thuật toán có thể nhầm các cặp này là near-duplicate.

Chạy:
    python inspect_near_duplicates.py --n 10
"""

import argparse
import csv
import math
import random
import matplotlib.pyplot as plt
from PIL import Image
from collections import defaultdict

REPORT_PATH = "../data/duplicate_report.csv"


def load_groups(report_path, leak_only=False):
    """
    Lọc các nhóm còn ở trạng thái CẦN REVIEW.
    Lưu ý: pipeline hiện tại đặt match_type = 'near_phash1', có thể kèm
    hậu tố '[LEAK - ƯU TIÊN REVIEW]' cho các trường hợp cần xem trước.
    Dùng .startswith('near_phash1') thay vì so khớp tuyệt đối 'near' (tên cũ).
    """
    groups = defaultdict(list)
    with open(report_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row["match_type"].startswith("near_phash1"):
                continue
            if leak_only and row["is_leak"].strip().lower() != "true":
                continue
            groups[row["group_id"]].append(row)
    return groups


def show_group(group_rows, group_id):
    n = len(group_rows)
    cols = min(n, 4)
    rows_n = math.ceil(n / cols)
    fig, axes = plt.subplots(rows_n, cols, figsize=(4 * cols, 4 * rows_n))
    axes = axes.flatten() if n > 1 else [axes]

    for i, row in enumerate(group_rows):
        try:
            img = Image.open(row["path"])
            axes[i].imshow(img)
            ssim_text = f"\nSSIM={row['ssim_score']}" if row.get("ssim_score") else ""
            axes[i].set_title(f"{row['split']}/{row['label']}\n{row['recommended_action']}{ssim_text}",
                               fontsize=9)
        except Exception as e:
            axes[i].set_title(f"Lỗi mở ảnh: {e}", fontsize=8)
        axes[i].axis("off")

    for j in range(n, len(axes)):
        axes[j].axis("off")

    has_leak = any("[LEAK" in r["match_type"] for r in group_rows)
    tag = " - LEAK, ƯU TIÊN XEM KỸ" if has_leak else ""
    fig.suptitle(f"Group {group_id} (near-duplicate, {n} ảnh){tag}")
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10,
                         help="Số nhóm near-duplicate muốn xem (mặc định 10)")
    parser.add_argument("--leak-only", action="store_true",
                         help="Chỉ xem các nhóm near-duplicate liên quan đến leak Train<->Test "
                              "(rủi ro cao nhất, nên ưu tiên kiểm tra trước)")
    parser.add_argument("--random", action="store_true",
                         help="Lấy mẫu NGẪU NHIÊN thay vì lấy N nhóm đầu tiên "
                              "(nên dùng khi số lượng nhóm lớn, để mẫu đại diện hơn)")
    parser.add_argument("--seed", type=int, default=42,
                         help="Seed cho random sample, để có thể chạy lại tái lập kết quả")
    args = parser.parse_args()

    groups = load_groups(REPORT_PATH, leak_only=args.leak_only)
    group_ids = list(groups.keys())

    label = "near-duplicate (LEAK Train<->Test)" if args.leak_only else "near-duplicate"
    print(f"Tổng số nhóm {label}: {len(group_ids)}")

    if args.random:
        random.seed(args.seed)
        selected_ids = random.sample(group_ids, min(args.n, len(group_ids)))
    else:
        selected_ids = group_ids[:args.n]

    print(f"Hiển thị {len(selected_ids)} nhóm (seed={args.seed})...\n")
    print("Đóng cửa sổ ảnh để xem nhóm tiếp theo.")

    for group_id in selected_ids:
        show_group(groups[group_id], group_id)


if __name__ == "__main__":
    main()