import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)

def evaluate(y_true, y_pred=None, y_prob=None, threshold=0.5, model_name="Model", plot_cm=True):
    """
    Hàm đánh giá mô hình dùng chung cho cả nhóm (Bài toán Bank Marketing mất cân bằng dữ liệu).

    Tham số:
    - y_true: Nhãn thực tế (0 hoặc 1).
    - y_pred: Nhãn dự đoán nhị phân (0 hoặc 1). Nếu truyền y_prob, y_pred sẽ tự tính theo threshold.
    - y_prob: Xác suất dự đoán thuộc lớp 1 (lấy từ model.predict_proba(X)[:, 1]). Bắt buộc để tính ROC-AUC & PR-AUC.
    - threshold: Ngưỡng phân loại (mặc định 0.5).
    - model_name: Tên mô hình hiển thị trên bảng kết quả và biểu đồ.
    - plot_cm: True nếu muốn vẽ kèm biểu đồ Ma trận nhầm lẫn (Confusion Matrix).

    Trả về:
    - dict chứa các chỉ số đánh giá để dễ dàng tổng hợp thành bảng so sánh.
    """
    if y_prob is not None:
        y_pred_eval = (np.asarray(y_prob) >= threshold).astype(int)
        roc_auc = roc_auc_score(y_true, y_prob)
        pr_auc = average_precision_score(y_true, y_prob)
    elif y_pred is not None:
        y_pred_eval = np.asarray(y_pred).astype(int)
        roc_auc = np.nan
        pr_auc = np.nan
    else:
        raise ValueError("Cần truyền ít nhất một trong hai tham số: 'y_prob' hoặc 'y_pred'.")

    precision = precision_score(y_true, y_pred_eval, zero_division=0)
    recall = recall_score(y_true, y_pred_eval, zero_division=0)
    f1 = f1_score(y_true, y_pred_eval, zero_division=0)
    cm = confusion_matrix(y_true, y_pred_eval)

    results = {
        "Model": model_name,
        "Threshold": threshold,
        "Precision": round(float(precision), 4),
        "Recall": round(float(recall), 4),
        "F1-Score": round(float(f1), 4),
        "ROC-AUC": round(float(roc_auc), 4) if not np.isnan(roc_auc) else None,
        "PR-AUC": round(float(pr_auc), 4) if not np.isnan(pr_auc) else None,
        "TN": int(cm[0, 0]),
        "FP": int(cm[0, 1]),
        "FN": int(cm[1, 0]),
        "TP": int(cm[1, 1])
    }

    print(f"=== KẾT QUẢ ĐÁNH GIÁ: {model_name} (Threshold = {threshold}) ===")
    print(f"Precision : {results['Precision']:.4f}")
    print(f"Recall    : {results['Recall']:.4f}")
    print(f"F1-Score  : {results['F1-Score']:.4f}")
    if y_prob is not None:
        print(f"ROC-AUC   : {results['ROC-AUC']:.4f}")
        print(f"PR-AUC    : {results['PR-AUC']:.4f}")
    print("\nChi tiết từng lớp:")
    print(classification_report(y_true, y_pred_eval, target_names=["No (0)", "Yes (1)"], digits=4))

    if plot_cm:
        plt.figure(figsize=(5, 4))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=["Dự đoán: No (0)", "Dự đoán: Yes (1)"],
            yticklabels=["Thực tế: No (0)", "Thực tế: Yes (1)"]
        )
        plt.title(f"Confusion Matrix - {model_name} (th={threshold})", fontweight="bold")
        plt.tight_layout()
        plt.show()

    return results