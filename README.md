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