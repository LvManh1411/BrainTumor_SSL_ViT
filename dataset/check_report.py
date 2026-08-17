import pandas as pd
df = pd.read_csv("../data/duplicate_report.csv")

to_delete = df[df['recommended_action'] == 'delete']

print("Tổng số ảnh đề xuất xóa:", len(to_delete))
print("\n--- Theo match_type ---")
print(to_delete['match_type'].value_counts())

print("\n--- Theo lý do (leak vs within-split) ---")
print(to_delete['is_leak'].value_counts())

print("\n--- Riêng phần LEAK (bắt buộc xóa) ---")
leak_delete = to_delete[to_delete['is_leak'] == True]
print(f"Số ảnh do leak: {len(leak_delete)}")
print(leak_delete['match_type'].value_counts())

leak_near = df[(df['is_leak']==True) & (df['match_type']=='near')]
print(leak_near['group_id'].unique())