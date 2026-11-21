# Từ điển dữ liệu & Quy ước bàn giao (UCI Bank Marketing)

## 1. Danh sách file bàn giao (`data/processed/` và `models/`)

| File | Kích thước | Mục đích & Đối tượng sử dụng |
| :--- | :--- | :--- |
| `train.parquet` & `test.parquet` | `32.940 × 65` / `8.236 × 65` | **Bản dữ liệu chính** (chia phân tầng `stratify=y`, bỏ `duration`). Dùng cho **TV2, TV3, TV4**. |
| `train_with_duration.parquet` & `test_with_duration.parquet` | `32.940 × 66` / `8.236 × 66` | **Bản tham chiếu** (giữ `duration`). Dùng cho **TV4** thí nghiệm tác động Data Leakage. |
| `train_time.parquet` & `test_time.parquet` | `32.940 × 65` / `8.236 × 65` | **Bản chia theo thời gian** (80% đầu làm train, 20% cuối làm test, không stratify). Dùng cho **TV2** kiểm tra độ bền mô hình. |
| `train_raw.parquet` & `test_raw.parquet` | `32.940 × 22` / `8.236 × 22` | Dữ liệu thô sau làm sạch (chưa One-Hot) để đối chiếu đầu vào. |
| `models/preprocessor.joblib` | Binary file | Đối tượng `ColumnTransformer` đã fit trên `train_raw` để **TV3** dùng cho app/demo dự đoán dữ liệu thô mới. |

---

## 2. Lưu ý kỹ thuật quan trọng cho các thành viên
- **Cho TV2 (Decision Tree & Kiểm tra độ bền):**
  - Chạy mô hình chính trên `train.parquet` / `test.parquet` (tỷ lệ `y=1` là `11.27%` ở cả 2 tập).
  - Chạy thêm kiểm tra độ bền trên `train_time.parquet` / `test_time.parquet`: lưu ý ở bản thời gian, tập train (2008–2009) chỉ có **~6.5%** nhãn `y=1`, trong khi tập test (2010) tăng vọt lên **~30.5%** do biến động lãi suất và chính sách gọi.
- **Cho TV3 (Demo / Ứng dụng):**
  - Nạp `preprocessor = joblib.load("models/preprocessor.joblib")` để biến đổi input thô từ người dùng. Lưu ý xóa cột `default_yes` (nếu có trong output của `preprocessor`) trước khi đưa vào `model.predict()`, khớp với danh sách cột trong `models/feature_metadata.json`.
- **Cho TV4 (Thí nghiệm mô hình tuyến tính / Logistic Regression):**
  - Các biến số trong file `.parquet` đang để nguyên thang đo gốc (`passthrough`, **chưa chuẩn hóa**). Nếu huấn luyện Logistic Regression hoặc SVM, cần bọc thêm `StandardScaler()` (fit trên tập train) và lưu ý nhóm biến vĩ mô (`emp.var.rate`, `euribor3m`, `nr.employed`) có tương quan rất cao ($r = 0.91 \text{--} 0.97$).

---

## 3. Chi tiết các quyết định xử lý & Đặc trưng
1. **Làm sạch:** Loại bỏ 12 dòng trùng lặp hoàn toàn (còn 41.176 dòng). Giữ nguyên giá trị ngoại lai (`campaign`, `duration`) vì mô hình dạng cây không nhạy cảm với ngoại lai.
2. **Xử lý `"unknown"`:** Giữ nguyên thành nhóm độc lập. Thực chứng từ dữ liệu cho thấy nhóm `default = unknown` có tỷ lệ đồng ý chỉ đạt **5.15%** (so với **12.88%** ở nhóm `default = no`), chứng minh trạng thái thiếu thông tin mang tín hiệu phân loại rõ rệt.
3. **Sửa logic cờ liên hệ trước đây:**
   - `was_contacted_before` (`0`/`1`): Xác định theo `previous > 0` (**5.625** khách hàng thực sự đã từng được liên hệ ở chiến dịch trước).
   - `has_valid_pdays` (`0`/`1`): Xác định theo `pdays != 999` (**1.515** khách hàng có ghi nhận số ngày cụ thể).
   - `pdays_clean`: Thay giá trị quy ước `999` bằng `-1`.
4. **Tránh trùng lặp đặc trưng:** Giữ nguyên `age` và `month` (One-Hot), không tạo thêm `age_group` hay `month_num` để tránh chia nhỏ độ quan trọng đặc trưng (`feature_importance`). Loại bỏ cột `default_yes` do gần như bằng 0 toàn bộ.