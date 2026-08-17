"""
02_apply_cleanup.py

Bước "Review -> Delete -> Final dataset" trong pipeline.

Đọc file duplicate_report.csv đã được xem lại (review), thực hiện xóa các
ảnh có quyết định cuối cùng là "delete", rồi in báo cáo số lượng ảnh cuối
cùng theo split/label.

QUY TẮC XÁC ĐỊNH QUYẾT ĐỊNH CUỐI CÙNG cho mỗi dòng trong CSV:
    - Nếu cột 'reviewer_decision' có giá trị ('keep' hoặc 'delete')
      -> DÙNG GIÁ TRỊ NÀY (người review đã tự quyết định, ưu tiên cao nhất)
    - Nếu 'reviewer_decision' để trống
      -> DÙNG 'recommended_action' (đề xuất mặc định của script)

Mặc định chạy ở chế độ DRY-RUN (chỉ in ra sẽ xóa gì, KHÔNG xóa thật).
Muốn xóa thật, chạy với cờ --confirm:

    python 02_apply_cleanup.py            # xem trước (dry-run)
    python 02_apply_cleanup.py --confirm  # xóa thật
"""

import os
import csv
import sys
from collections import defaultdict

REPORT_PATH = "../data/duplicate_report.csv"
PROCESSED_DATA_DIR = "../data/processed"
VALID_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")


def read_report(report_path):
    if not os.path.exists(report_path):
        raise FileNotFoundError(
            f"Không tìm thấy {report_path}. Hãy chạy 01_build_report.py trước."
        )
    with open(report_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return list(reader)


def resolve_final_decision(row):
    """
    Ưu tiên reviewer_decision nếu có (người review đã tự quyết định).
    Nếu không có:
        - recommended_action = 'delete' -> xóa (trường hợp exact, độ tin cậy cao)
        - recommended_action = 'keep'   -> giữ
        - recommended_action = 'review' -> MẶC ĐỊNH GIỮ LẠI (an toàn), vì đây
          là near-duplicate CHƯA được xác nhận thủ công. Chỉ xóa khi người
          review tự điền reviewer_decision = 'delete' sau khi xem ảnh + SSIM.
    """
    reviewer = row.get("reviewer_decision", "").strip().lower()
    if reviewer in ("keep", "delete"):
        return reviewer

    action = row["recommended_action"].strip().lower()
    if action == "review":
        return "keep"  # an toàn: chưa review thì không xóa
    return action


def apply_cleanup(rows, dry_run=True):
    to_delete = []
    n_pending_review = 0

    for row in rows:
        decision = resolve_final_decision(row)
        reviewer = row.get("reviewer_decision", "").strip().lower()
        action = row["recommended_action"].strip().lower()

        if action == "review" and reviewer not in ("keep", "delete"):
            n_pending_review += 1

        if decision == "delete":
            to_delete.append(row)
        elif decision not in ("keep", "delete"):
            print(f"[Cảnh báo] Giá trị không hợp lệ ở dòng path={row['path']}: "
                  f"'{decision}' -> mặc định giữ lại (keep)")

    print(f"Số ảnh sẽ bị xóa: {len(to_delete)} / {len(rows)}")
    if n_pending_review > 0:
        print(f"[Lưu ý] Còn {n_pending_review} ảnh ở trạng thái 'review' CHƯA được "
              f"quyết định (reviewer_decision để trống) -> đang được GIỮ LẠI theo "
              f"mặc định an toàn. Nếu muốn xóa, hãy điền 'delete' vào cột "
              f"reviewer_decision trong {REPORT_PATH} rồi chạy lại.")

    if dry_run:
        print("\n[DRY-RUN] Chưa xóa gì cả. Một vài ví dụ ảnh sẽ bị xóa nếu chạy --confirm:")
        for row in to_delete[:15]:
            print(f"  - [{row['split']}/{row['label']}] {row['path']}")
        if len(to_delete) > 15:
            print(f"  ... và {len(to_delete) - 15} ảnh khác")
        print("\nChạy lại với --confirm để thực sự xóa: python 02_apply_cleanup.py --confirm")
        return 0

    n_deleted, n_missing = 0, 0
    for row in to_delete:
        path = row["path"]
        if os.path.exists(path):
            os.remove(path)
            n_deleted += 1
        else:
            n_missing += 1

    print(f"Đã xóa: {n_deleted} ảnh" + (f" | Không tìm thấy (đã xóa trước đó?): {n_missing}" if n_missing else ""))
    return n_deleted


def report_final_counts(processed_dir):
    print("\n" + "=" * 60)
    print("DATASET CUỐI CÙNG (Final dataset)")
    print("=" * 60)
    total = 0
    counts = defaultdict(lambda: defaultdict(int))
    for split in ["Training", "Testing"]:
        split_path = os.path.join(processed_dir, split)
        if not os.path.exists(split_path):
            continue
        for label in sorted(os.listdir(split_path)):
            label_path = os.path.join(split_path, label)
            if not os.path.isdir(label_path):
                continue
            n = len([f for f in os.listdir(label_path) if f.lower().endswith(VALID_EXTENSIONS)])
            counts[split][label] = n
            total += n

    for split in ["Training", "Testing"]:
        for label, n in counts[split].items():
            print(f"  {split}/{label}: {n}")
    print(f"\nTổng số ảnh cuối cùng: {total}")


def main():
    dry_run = "--confirm" not in sys.argv

    print("=" * 60)
    print(f"Đọc báo cáo: {REPORT_PATH}")
    print("=" * 60)
    rows = read_report(REPORT_PATH)
    print(f"Tổng số dòng trong báo cáo: {len(rows)}")

    print("\n" + "=" * 60)
    print(f"Áp dụng quyết định xóa (chế độ: {'DRY-RUN' if dry_run else 'XÓA THẬT'})")
    print("=" * 60)
    apply_cleanup(rows, dry_run=dry_run)

    if not dry_run:
        report_final_counts(PROCESSED_DATA_DIR)
        print("\nHoàn tất. Dữ liệu trong data/processed/ đã sẵn sàng cho bước huấn luyện SSL/ViT.")


if __name__ == "__main__":
    main()