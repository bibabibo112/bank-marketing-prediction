# Dự đoán khả năng đăng ký tiền gửi có kỳ hạn (UCI Bank Marketing)

Dự án Phân tích Dữ liệu Lớn dự đoán khách hàng đăng ký tiền gửi có kỳ hạn (`y = 1`) trên bộ dữ liệu **UCI Bank Marketing** (`bank-additional-full.csv`), cố định `RANDOM_STATE = 42`.

## 1. Cấu trúc thư mục
```text
├── data/
│   ├── raw/                  # Dữ liệu gốc bank-additional-full.csv
│   └── processed/            # Dữ liệu sạch (.parquet) & data_dictionary.md
├── models/                   # preprocessor.joblib & feature_metadata.json
├── notebooks/                # Các notebook phân tích & huấn luyện theo thứ tự
├── reports/figures/          # Biểu đồ xuất ra phục vụ báo cáo
├── src/
│   └── metrics.py            # Hàm evaluate() dùng chung cho cả nhóm
└── requirements.txt          # Danh sách thư viện chuẩn
```

---

## 5. Phân công nhiệm vụ các thành viên tiếp theo

### Thành viên 2: Decision Tree, Random Forest & Tối ưu ngưỡng kinh doanh (`notebooks/02_decision_tree_rf.ipynb`)
1. Đọc `data/processed/train.parquet` và `test.parquet` (bản chính, không có `duration`).
2. Huấn luyện **Baseline** Decision Tree & Random Forest với tham số mặc định, ghi lại kết quả bằng `src.metrics.evaluate()`.
3. Tinh chỉnh siêu tham số bằng **Stratified Cross-Validation chỉ trên tập `train`** (`max_depth`, `min_samples_leaf`, `n_estimators`, `class_weight`...). Tối ưu theo **PR-AUC hoặc F1** (không dùng Accuracy).
4. Đánh giá trên `test`: Precision, Recall, F1, ROC-AUC, PR-AUC, Confusion Matrix, đường cong ROC và PR.
5. **Chọn ngưỡng kinh doanh:** Đặt giả định chi phí mỗi cuộc gọi và doanh thu mỗi khách đăng ký; quét ngưỡng `0.05` $\rightarrow$ `0.95` trên tập validation / out-of-fold của `train` để tìm ngưỡng tối ưu lợi nhuận, sau đó mới áp dụng ngưỡng đó báo cáo trên `test`.
6. Lập bảng kịch bản ngưỡng (số cuộc gọi, số khách chốt, lợi nhuận) và rút trích `feature_importance`.
7. Chạy lại mô hình tốt nhất trên `train_time.parquet` / `test_time.parquet` để kiểm tra độ bền theo thời gian.
8. Viết **Mục 4** trong báo cáo (Mô hình cây và tối ưu ngưỡng).

### Thành viên 3: PySpark MLlib, Demo & Quản lý Repo (`notebooks/03_spark_mllib.ipynb` & `app/`)
1. Đọc file `.parquet` bằng PySpark; dùng `VectorAssembler` gom các cột đặc trưng (dữ liệu đã One-Hot nên không cần `StringIndexer`).
2. Huấn luyện Decision Tree và Random Forest bằng **Spark MLlib**, đối chiếu kết quả với bản scikit-learn của TV2 (cùng split) để xác nhận pipeline hoạt động đúng.
3. **Thí nghiệm mở rộng (Scalability):** Nhân bản tập `train` ở các mức `1x`, `10x`, `50x`, `100x`, đo thời gian huấn luyện và mức tiêu thụ RAM (chạy lặp lại nhiều lần lấy trung bình; thảo luận trung thực về overhead của Spark ở chế độ local với dữ liệu nhỏ).
4. **Dựng Demo (Notebook hoặc Streamlit):** Nạp `models/preprocessor.joblib` + mô hình tốt nhất; cho phép nhập thông tin khách hàng thô, trả về xác suất và quyết định gọi/không gọi theo ngưỡng tối ưu của TV2 (lưu ý xóa cột `default_yes` nếu có trước khi `predict` để khớp `models/feature_metadata.json`).
5. Quản lý nhánh GitHub, giữ ổn định phiên bản Python chung của nhóm (3.11 hoặc 3.12).
6. Viết **Mục 2** (Cơ sở lý thuyết) và phần kiến trúc Spark trong **Mục 3**.

### Thành viên 4: Baseline tuyến tính, Rò rỉ `duration` & Xử lý mất cân bằng (`notebooks/04_experiments_comparison.ipynb`)
1. **Baseline đối chứng:** Chạy mô hình ngây thơ (luôn dự đoán `"no"`) để chỉ ra sự sai lệch của Accuracy; chạy **Logistic Regression** (bắt buộc bọc `StandardScaler` chỉ `fit` trên `train` và cân nhắc loại bớt biến vĩ mô tương quan $>0.9$ như `euribor3m`, `nr.employed`).
2. **Thí nghiệm Data Leakage:** Huấn luyện cùng một cấu hình mô hình trên `train.parquet` và `train_with_duration.parquet`, so sánh bằng cùng bộ chỉ số để chứng minh mức độ sai lệch do `duration` gây ra.
3. **Thí nghiệm mất cân bằng:** So sánh `class_weight`, Oversampling (SMOTE/RandomOverSampler) và Undersampling. **Bắt buộc** đặt bước resampling bên trong `imblearn.Pipeline` và nằm trong vòng Cross-Validation của `train` (tuyệt đối không resample trước khi chia tập).
4. Phân tích `feature_importance` và chuẩn hóa style vẽ biểu đồ (ROC, PR, Confusion Matrix) thống nhất cho toàn bộ báo cáo.
5. Phối hợp cùng TV3 chạy thí nghiệm so sánh hiệu năng Spark và đơn máy.
6. Viết **Mục 5** (Thảo luận).

---

## 6. Quy tắc làm việc chung (Cả nhóm)
- **Git Workflow:** Mỗi thành viên tạo nhánh riêng (ví dụ: `feat/tv2-tree-models`, `feat/tv3-spark-demo`, `feat/tv4-experiments`), không làm việc trực tiếp hay ghi đè notebook của người khác, merge vào `main` qua Pull Request.
- **Thống nhất đánh giá:** Mọi bảng so sánh mô hình đều phải chạy trên cùng bộ split (`data/processed/`), cùng `RANDOM_STATE = 42` và cùng gọi hàm `evaluate()` trong `src/metrics.py`.
- **Đồng bộ số liệu:** Tổ chức họp ngắn ngay sau khi TV2 và TV4 có kết quả chạy đầu tiên để chốt bảng số liệu chung trước khi viết báo cáo.
- **Hoàn thiện cuối kỳ:** Ghép Mục 6, Mục 7, xuất báo cáo PDF và làm slide thuyết trình.

## Chạy lại thí nghiệm TV4

Chọn kernel Python 3.12 đã cài `requirements.txt`, rồi chạy `notebooks/04_experiments_comparison.ipynb`. Bảng kết quả và mục thảo luận nằm trong `reports/tv4`. Hình TV4 trong `reports/figures` dùng nền trắng, palette colorblind và 300 dpi.

Chạy lần lượt các lệnh benchmark từ thư mục gốc. Khuyến nghị dùng JDK 17 hoặc 21 cho PySpark.

```powershell
python -m pip install psutil "pyspark>=4.1,<4.2"
python benchmark_tv4.py scikit-learn
python benchmark_tv4.py Spark
```

Chạy lại ô Benchmark của notebook để cập nhật bảng và hình từ `reports/tv4/benchmark_runs.csv`.
