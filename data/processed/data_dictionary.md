# Từ điển dữ liệu & Quy ước bàn giao (UCI Bank Marketing)

## 1. Danh sách file bàn giao (`data/processed/` và `models/`)
Toàn bộ dữ liệu đã được làm sạch, cố định tập phân chia Train/Test (`random_state = 42`) và lưu dưới định dạng `.parquet`:

| Tên file | Kích thước | Mục đích & Đối tượng sử dụng |
| :--- | :--- | :--- |
| `train.parquet` & `test.parquet` | `32.940 × 65` / `8.236 × 65` | **Bản dữ liệu chính** (chia phân tầng `stratify=y`, đã bỏ `duration`). Dùng cho **TV2, TV3, TV4** huấn luyện mô hình chính. |
| `train_with_duration.parquet` & `test_with_duration.parquet` | `32.940 × 66` / `8.236 × 66` | **Bản tham chiếu** (cùng tập chia chính nhưng giữ `duration`). Dùng cho **TV4** chạy thí nghiệm đối chứng tác động của Data Leakage. |
| `train_time.parquet` & `test_time.parquet` | `32.940 × 65` / `8.236 × 65` | **Bản chia theo thời gian** (80% dòng đầu làm train, 20% dòng cuối làm test, không phân tầng). Dùng cho **TV2** kiểm tra độ bền mô hình. |
| `train_raw.parquet` & `test_raw.parquet` | `32.940 × 22` / `8.236 × 22` | Dữ liệu sau làm sạch và tạo đặc trưng mức dòng nhưng **chưa mã hóa One-Hot**. Dùng để đối chiếu đầu vào thô. |
| `models/preprocessor.joblib` | Binary file | Đối tượng `ColumnTransformer` đã `fit` trên tập `train_raw` để **TV3** nạp vào ứng dụng/demo dự đoán dữ liệu thô mới. |
| `models/feature_metadata.json` | JSON file | Lưu danh sách tên cột đầu vào thô và thứ tự các cột đặc trưng chuẩn sau khi biến đổi. |

### Đoạn code mẫu nạp dữ liệu chuẩn:
```python
import pandas as pd

train_df = pd.read_parquet("data/processed/train.parquet")
test_df = pd.read_parquet("data/processed/test.parquet")

X_train, y_train = train_df.drop(columns=["y"]), train_df["y"]
X_test, y_test = test_df.drop(columns=["y"]), test_df["y"]