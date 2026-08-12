from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": cm
    }

# ---  TEST THỬ HÀM ---
if __name__ == "__main__":
    # Tạo nhãn thực tế và nhãn mô hình đoán thử (10 mẫu)
    y_true = [0, 1, 2, 3, 0, 1, 2, 3, 0, 1]
    y_pred = [0, 1, 2, 3, 0, 1, 2, 0, 0, 1]  # Đoán đúng 9/10
    
    results = compute_metrics(y_true, y_pred)
    
    print(" Kết quả chạy thử module metrics.py:")
    print(f" Accuracy : {results['accuracy']:.4f}")
    print(f" Precision: {results['precision']:.4f}")
    print(f" Recall   : {results['recall']:.4f}")
    print(f" F1-Score : {results['f1_score']:.4f}")
    print(" Confusion Matrix:")
    print(results['confusion_matrix'])