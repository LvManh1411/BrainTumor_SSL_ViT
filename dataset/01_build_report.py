"""
01_build_report.py

Pipeline tiền xử lý dữ liệu (Nội dung 1 - thuyết minh đề tài):

    RAW
     |
     v
    Validate image        (kiểm tra file không hỏng, đọc được)
     |
     v
    Convert RGB            (đồng nhất mode: L / RGBA / P -> RGB, tránh đụng độ tên file)
     |
     v
    Exact duplicate        (MD5 hash - trùng tuyệt đối theo nội dung byte)
     |
     v
    pHash                  (Perceptual hash - phát hiện ảnh gần giống nhau)
     |
     v
    Near-duplicate          (so khớp phash trong ngưỡng cho phép, CHỈ trong cùng label
                             để giảm chi phí tính toán - ảnh khác lớp u không thể là
                             near-duplicate của nhau)
     |
     v
    Union-Find grouping      (gom nhóm trùng lặp đúng theo tính bắc cầu)
     |
     v
    Detect Train <-> Test leakage
     |
     v
    Generate duplicate_report.csv     <-- SCRIPT NÀY DỪNG Ở ĐÂY, KHÔNG TỰ XÓA

Sau khi có report, xem lại (Review) bằng tay hoặc Excel, rồi chạy
02_apply_cleanup.py để thực sự xóa ảnh theo quyết định cuối cùng.

Yêu cầu cài thêm thư viện:
    pip install imagehash pandas tqdm

Chạy:
    python 01_build_report.py
"""

import os
import hashlib
import csv
from collections import defaultdict

import numpy as np
from PIL import Image
import imagehash
from skimage.metrics import structural_similarity as ssim

try:
    from tqdm import tqdm
except ImportError:
    # Nếu chưa cài tqdm, dùng hàm giả để code vẫn chạy được (không có progress bar)
    def tqdm(iterable, **kwargs):
        return iterable


# ==== CẤU HÌNH ====
RAW_DATA_DIR = "../data/raw"
PROCESSED_DATA_DIR = "../data/processed"
REPORT_PATH = "../data/duplicate_report.csv"

VALID_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")

# Ngưỡng khoảng cách Hamming giữa 2 pHash để tạo liên kết trong Union-Find.
# threshold=1 nghĩa là chấp nhận union các ảnh cách nhau 0 hoặc 1.
# Sau khi gom nhóm, mỗi nhóm sẽ được PHÂN LOẠI LẠI theo khoảng cách lớn
# nhất trong nhóm (xem classify_group_match_type):
#   - MD5 giống hệt                    -> match_type = 'exact_md5'   (tự động)
#   - pHash cách nhau = 0 (mọi cặp)     -> match_type = 'exact_phash0' (tự động)
#   - Có cặp cách nhau = 1              -> match_type = 'near_phash1'
#         -> BẮT BUỘC tính thêm SSIM và để 'review' (KHÔNG tự xóa),
#            chờ xác nhận thủ công trước khi coi là delete thật sự.
HASH_DISTANCE_THRESHOLD = 1

# Kích thước resize khi tính SSIM (đưa 2 ảnh về cùng kích thước để so sánh)
SSIM_RESIZE = (256, 256)


# =========================================================
# BƯỚC 1: Quét + Validate ảnh trong RAW
# =========================================================

def collect_image_paths(base_dir):
    """Duyệt toàn bộ ảnh theo cấu trúc split/label/*.jpg"""
    records = []
    for split in ["Training", "Testing"]:
        split_path = os.path.join(base_dir, split)
        if not os.path.exists(split_path):
            print(f"[Cảnh báo] Không tìm thấy thư mục: {split_path}")
            continue
        for label in os.listdir(split_path):
            label_path = os.path.join(split_path, label)
            if not os.path.isdir(label_path):
                continue
            for file in os.listdir(label_path):
                if file.lower().endswith(VALID_EXTENSIONS):
                    records.append({
                        "path": os.path.join(label_path, file),
                        "split": split,
                        "label": label,
                        "filename": file,
                    })
    return records


def validate_images(records):
    """Loại bỏ ảnh hỏng (0 byte, không giải mã được)."""
    valid_records = []
    n_corrupted = 0

    for rec in tqdm(records, desc="Validate images"):
        path = rec["path"]
        if os.path.getsize(path) == 0:
            print(f"[Lỗi] File rỗng: {path}")
            n_corrupted += 1
            continue
        try:
            with Image.open(path) as img:
                img.verify()
            with Image.open(path) as img:
                img.load()
            valid_records.append(rec)
        except Exception as e:
            print(f"[Lỗi] File hỏng: {path} | {e}")
            n_corrupted += 1

    print(f"Ảnh hợp lệ: {len(valid_records)} | Ảnh hỏng bị loại: {n_corrupted}")
    return valid_records


# =========================================================
# BƯỚC 2: Convert RGB, lưu sang processed/ (tránh đụng độ tên file)
# =========================================================

def convert_to_rgb(valid_records, processed_dir):
    """
    Convert từng ảnh sang RGB và lưu ra processed_dir, giữ nguyên cấu trúc
    split/label. Kiểm tra đụng độ tên file trước khi ghi - nếu 2 ảnh khác
    nội dung nhưng trùng tên (hay gặp khi dataset gộp từ nhiều nguồn),
    ảnh sau sẽ được đổi tên thay vì ghi đè âm thầm lên ảnh trước.
    """
    processed_records = []
    n_saved, n_failed, n_renamed = 0, 0, 0
    used_paths_per_dir = defaultdict(set)

    for rec in tqdm(valid_records, desc="Convert RGB"):
        dst_dir = os.path.join(processed_dir, rec["split"], rec["label"])
        os.makedirs(dst_dir, exist_ok=True)

        filename = rec["filename"]
        dst_path = os.path.join(dst_dir, filename)

        if dst_path in used_paths_per_dir[dst_dir] or os.path.exists(dst_path):
            name, ext = os.path.splitext(filename)
            counter = 1
            while True:
                new_filename = f"{name}_dup{counter}{ext}"
                new_dst_path = os.path.join(dst_dir, new_filename)
                if new_dst_path not in used_paths_per_dir[dst_dir] and not os.path.exists(new_dst_path):
                    break
                counter += 1
            filename, dst_path = new_filename, new_dst_path
            n_renamed += 1

        used_paths_per_dir[dst_dir].add(dst_path)

        try:
            with Image.open(rec["path"]) as img:
                img = img.convert("RGB")
                img.save(dst_path)
            processed_records.append({
                "path": dst_path,
                "split": rec["split"],
                "label": rec["label"],
                "filename": filename,
            })
            n_saved += 1
        except Exception as e:
            print(f"[Lỗi] Không xử lý được ảnh: {rec['path']} | {e}")
            n_failed += 1

    print(f"Đã lưu: {n_saved} ảnh | Lỗi: {n_failed} | Đổi tên do đụng độ: {n_renamed}")
    return processed_records


# =========================================================
# BƯỚC 3: Exact duplicate (MD5) + pHash
# =========================================================

def compute_md5(filepath, chunk_size=8192):
    md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        while chunk := f.read(chunk_size):
            md5.update(chunk)
    return md5.hexdigest()


def compute_hashes(records):
    """Tính cả MD5 (exact) và pHash (near-duplicate) cho từng ảnh."""
    for rec in tqdm(records, desc="Compute MD5 + pHash"):
        try:
            rec["md5"] = compute_md5(rec["path"])
            with Image.open(rec["path"]) as img:
                rec["phash"] = imagehash.phash(img)
        except Exception as e:
            print(f"[Lỗi] Không tính được hash: {rec['path']} | {e}")
            rec["md5"], rec["phash"] = None, None
    return [r for r in records if r["phash"] is not None]


# =========================================================
# BƯỚC 4: Union-Find grouping (Exact + Near-duplicate)
# =========================================================

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx != ry:
            self.parent[rx] = ry


def group_duplicates(records, threshold=HASH_DISTANCE_THRESHOLD):
    """
    Gom nhóm trùng lặp bằng Union-Find, đảm bảo tính bắc cầu:
        1. Union các ảnh có MD5 giống hệt nhau (exact duplicate) - toàn dataset.
        2. Union các ảnh có pHash cách nhau <= threshold (near-duplicate) -
           CHỈ so trong cùng label (ảnh khác lớp u não không thể là near-dup
           của nhau, giới hạn này giúp giảm số phép so sánh đáng kể).
    """
    n = len(records)
    uf = UnionFind(n)

    # --- Union theo MD5 (exact duplicate, toàn dataset) ---
    md5_groups = defaultdict(list)
    for i, rec in enumerate(records):
        if rec["md5"] is not None:
            md5_groups[rec["md5"]].append(i)
    for idx_list in md5_groups.values():
        for i in idx_list[1:]:
            uf.union(idx_list[0], i)

    # --- Union theo pHash (near-duplicate, trong cùng label) ---
    label_buckets = defaultdict(list)
    for i, rec in enumerate(records):
        label_buckets[rec["label"]].append(i)

    for label, idx_list in tqdm(label_buckets.items(), desc="Near-duplicate theo label"):
        m = len(idx_list)
        for a in range(m):
            for b in range(a + 1, m):
                i, j = idx_list[a], idx_list[b]
                if uf.find(i) == uf.find(j):
                    continue  # đã cùng nhóm rồi, khỏi so lại
                dist = records[i]["phash"] - records[j]["phash"]
                if dist <= threshold:
                    uf.union(i, j)

    # --- Gom kết quả thành các nhóm cuối cùng ---
    groups = defaultdict(list)
    for i in range(n):
        root = uf.find(i)
        groups[root].append(i)

    return list(groups.values())


def classify_group_match_type(group_records):
    """
    Phân loại lại 1 nhóm dựa trên khoảng cách pHash LỚN NHẤT giữa các cặp
    trong nhóm (không chỉ dựa vào MD5 đơn thuần như trước):

        - Toàn bộ ảnh trong nhóm CÙNG MD5              -> 'exact_md5'
        - MD5 khác nhau nhưng MỌI cặp pHash cách nhau 0 -> 'exact_phash0'
          (khác byte do nén/resize lại, nhưng nội dung giống hệt tuyệt đối
           về mặt cảm quan -> đủ tin cậy để tự động xử lý)
        - Có ÍT NHẤT 1 cặp pHash cách nhau = 1          -> 'near_phash1'
          (CHƯA đủ chắc chắn -> cần SSIM + xác nhận thủ công)
    """
    md5_set = set(r["md5"] for r in group_records if r["md5"] is not None)
    if len(md5_set) <= 1:
        return "exact_md5"

    max_dist = 0
    n = len(group_records)
    for a in range(n):
        for b in range(a + 1, n):
            dist = group_records[a]["phash"] - group_records[b]["phash"]
            max_dist = max(max_dist, dist)

    return "exact_phash0" if max_dist == 0 else "near_phash1"


def compute_ssim_score(path_a, path_b, resize=SSIM_RESIZE):
    """Tính điểm SSIM (0-1, càng gần 1 càng giống nhau) giữa 2 ảnh."""
    try:
        img_a = Image.open(path_a).convert("L").resize(resize)
        img_b = Image.open(path_b).convert("L").resize(resize)
        arr_a = np.array(img_a)
        arr_b = np.array(img_b)
        score, _ = ssim(arr_a, arr_b, full=True)
        return round(float(score), 4)
    except Exception as e:
        print(f"[Lỗi] Không tính được SSIM giữa {path_a} và {path_b}: {e}")
        return None


# =========================================================
# BƯỚC 5: Phát hiện leak Train <-> Test + quyết định đề xuất
# =========================================================

def build_report_rows(records, groups):
    """
    Với mỗi nhóm trùng lặp, xác định match_type (xem classify_group_match_type),
    is_leak (nhóm có mặt ở cả Training và Testing), và recommended_action.

    QUY TẮC ĐỀ XUẤT (đã nâng cấp - phân biệt rõ mức độ tin cậy):

    1) match_type = 'exact_md5' hoặc 'exact_phash0'  (ĐỘ TIN CẬY CAO)
        - Xử lý TỰ ĐỘNG như trước:
            + Nếu leak (cả Train + Test): giữ 1 bản Training, xóa phần còn lại
              (kể cả trùng lặp nội bộ trong Training)
            + Nếu không leak: giữ ảnh đầu tiên trong split, xóa phần còn lại

    2) match_type = 'near_phash1'  (CHƯA CHẮC CHẮN - cần xác minh thêm)
        - KHÔNG tự động đề xuất 'delete'. Thay vào đó:
            + Tính SSIM giữa ảnh tham chiếu (ảnh đầu tiên trong nhóm) và
              từng ảnh còn lại, ghi vào cột 'ssim_score'
            + recommended_action = 'review' cho các ảnh không phải tham chiếu
              -> Ở bước apply_cleanup, 'review' mặc định được xử lý AN TOÀN
                 là 'keep' (không xóa) trừ khi người review tự điền
                 reviewer_decision = 'delete' sau khi xem ảnh + điểm SSIM.
            + Nếu nhóm này còn là leak (Train <-> Test), gắn cờ đặc biệt
              trong ghi chú để người review biết đây là trường hợp QUAN
              TRỌNG cần ưu tiên xem trước (có thể là leak thật).
    """
    rows = []
    for group_id, idx_list in enumerate(groups):
        group_records = [records[i] for i in idx_list]
        splits_in_group = set(r["split"] for r in group_records)
        is_leak = len(splits_in_group) > 1

        if len(group_records) == 1:
            rec = group_records[0]
            rows.append({
                "group_id": group_id,
                "path": rec["path"],
                "split": rec["split"],
                "label": rec["label"],
                "md5": rec["md5"],
                "phash": str(rec["phash"]),
                "group_size": 1,
                "match_type": "unique",
                "ssim_score": "",
                "is_leak": False,
                "recommended_action": "keep",
                "reviewer_decision": "",
            })
            continue

        match_type = classify_group_match_type(group_records)

        # ---- Trường hợp ĐỘ TIN CẬY CAO: xử lý tự động như cũ ----
        if match_type in ("exact_md5", "exact_phash0"):
            if is_leak:
                training_records = [r for r in group_records if r["split"] == "Training"]
                kept_path = training_records[0]["path"] if training_records else group_records[0]["path"]
            else:
                kept_path = group_records[0]["path"]

            for rec in group_records:
                action = "keep" if rec["path"] == kept_path else "delete"
                rows.append({
                    "group_id": group_id,
                    "path": rec["path"],
                    "split": rec["split"],
                    "label": rec["label"],
                    "md5": rec["md5"],
                    "phash": str(rec["phash"]),
                    "group_size": len(group_records),
                    "match_type": match_type,
                    "ssim_score": "",
                    "is_leak": is_leak,
                    "recommended_action": action,
                    "reviewer_decision": "",
                })

        # ---- Trường hợp CHƯA CHẮC CHẮN: bắt buộc SSIM + review ----
        else:  # near_phash1
            # Ưu tiên chọn ảnh Training làm tham chiếu nếu có (để việc xóa
            # tập trung vào Testing khi thật sự confirm là leak)
            training_records = [r for r in group_records if r["split"] == "Training"]
            reference = training_records[0] if training_records else group_records[0]

            for rec in group_records:
                if rec["path"] == reference["path"]:
                    rows.append({
                        "group_id": group_id,
                        "path": rec["path"],
                        "split": rec["split"],
                        "label": rec["label"],
                        "md5": rec["md5"],
                        "phash": str(rec["phash"]),
                        "group_size": len(group_records),
                        "match_type": match_type,
                        "ssim_score": "",
                        "is_leak": is_leak,
                        "recommended_action": "keep",
                        "reviewer_decision": "",
                    })
                else:
                    ssim_score = compute_ssim_score(reference["path"], rec["path"])
                    note_leak = " [LEAK - ƯU TIÊN REVIEW]" if (is_leak and rec["split"] == "Testing") else ""
                    rows.append({
                        "group_id": group_id,
                        "path": rec["path"],
                        "split": rec["split"],
                        "label": rec["label"],
                        "md5": rec["md5"],
                        "phash": str(rec["phash"]),
                        "group_size": len(group_records),
                        "match_type": match_type + note_leak,
                        "ssim_score": ssim_score,
                        "is_leak": is_leak,
                        "recommended_action": "review",  # KHÔNG tự xóa
                        "reviewer_decision": "",
                    })

    return rows


def write_report_csv(rows, report_path):
    fieldnames = ["group_id", "path", "split", "label", "md5", "phash",
                  "group_size", "match_type", "ssim_score", "is_leak",
                  "recommended_action", "reviewer_decision"]
    with open(report_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Đã ghi báo cáo: {report_path}")


# =========================================================
# MAIN
# =========================================================

def main():
    print("=" * 60)
    print("BƯỚC 1: Quét ảnh trong RAW")
    print("=" * 60)
    raw_records = collect_image_paths(RAW_DATA_DIR)
    print(f"Tổng số ảnh tìm thấy: {len(raw_records)}")

    print("\n" + "=" * 60)
    print("BƯỚC 2: Validate image")
    print("=" * 60)
    valid_records = validate_images(raw_records)

    print("\n" + "=" * 60)
    print("BƯỚC 3: Convert RGB -> processed/")
    print("=" * 60)
    processed_records = convert_to_rgb(valid_records, PROCESSED_DATA_DIR)

    print("\n" + "=" * 60)
    print("BƯỚC 4: Exact duplicate (MD5) + pHash")
    print("=" * 60)
    processed_records = compute_hashes(processed_records)

    print("\n" + "=" * 60)
    print("BƯỚC 5: Near-duplicate + Union-Find grouping")
    print("=" * 60)
    groups = group_duplicates(processed_records, HASH_DISTANCE_THRESHOLD)
    n_dup_groups = sum(1 for g in groups if len(g) > 1)
    n_leak_groups = sum(
        1 for g in groups
        if len(g) > 1 and len(set(processed_records[i]["split"] for i in g)) > 1
    )
    print(f"Tổng số nhóm: {len(groups)} | Nhóm có trùng lặp (size > 1): {n_dup_groups}")
    print(f"Trong đó, nhóm bị LEAK Training<->Testing: {n_leak_groups}")

    print("\n" + "=" * 60)
    print("BƯỚC 6: Generate duplicate_report.csv")
    print("=" * 60)
    rows = build_report_rows(processed_records, groups)
    write_report_csv(rows, REPORT_PATH)

    n_delete = sum(1 for r in rows if r["recommended_action"] == "delete")
    n_review = sum(1 for r in rows if r["recommended_action"] == "review")
    print(f"\nSố ảnh đề xuất XÓA TỰ ĐỘNG (exact, độ tin cậy cao): {n_delete} / {len(rows)}")
    print(f"Số ảnh cần REVIEW THỦ CÔNG (near-duplicate, chưa chắc chắn): {n_review} / {len(rows)}")
    print(f"  -> Các ảnh 'review' mặc định sẽ được GIỮ LẠI (an toàn) ở bước")
    print(f"     apply_cleanup, trừ khi bạn tự điền reviewer_decision='delete'")
    print(f"     sau khi xem ảnh + điểm SSIM trong report.")
    print("\n>>> DỪNG LẠI Ở ĐÂY - CHƯA XÓA GÌ CẢ <<<")
    print(f"Vui lòng mở file {REPORT_PATH} để REVIEW (xem lại) trước khi xóa.")
    print("Ưu tiên xem các dòng có match_type chứa '[LEAK - ƯU TIÊN REVIEW]' trước.")
    print("Sau khi review xong, chạy: python 02_apply_cleanup.py")


if __name__ == "__main__":
    main()