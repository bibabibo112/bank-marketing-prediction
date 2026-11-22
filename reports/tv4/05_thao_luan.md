# 5. Thảo luận

Các phép so sánh dùng đúng bộ Parquet của TV2, gồm 32.940 dòng train và 8.236 dòng test. Có 928 khách đăng ký trong tập test (11,27%). Notebook huấn luyện lại hai mô hình cây với seed 42 và siêu tham số công bố tại commit `caface4` của `feat/tv2-tree-models`. PR-AUC là Average Precision (AP) theo hàm đánh giá chung. Bảng so sánh mặc định dùng ngưỡng 0,5.

## 5.1 Baseline và mô hình cây

Mô hình luôn dự đoán `no` đạt Accuracy 88,73% dù bỏ sót cả 928 khách đăng ký. Recall và F1 của lớp `yes` đều bằng 0, còn AP chỉ bằng tỷ lệ khách đăng ký (0,1127). Vì vậy AP và các chỉ số của lớp `yes` phù hợp hơn khi đánh giá khả năng lập danh sách gọi.

| Mô hình bỏ `duration` | Precision | Recall | F1 | ROC-AUC | AP |
| --- | ---: | ---: | ---: | ---: | ---: |
| Luôn dự đoán `no` | 0,0000 | 0,0000 | 0,0000 | 0,5000 | 0,1127 |
| Logistic Regression | 0,6502 | 0,2123 | 0,3201 | 0,8005 | 0,4536 |
| Decision Tree tinh chỉnh | 0,6687 | 0,2349 | 0,3477 | 0,8030 | 0,4686 |
| Random Forest tinh chỉnh | 0,6746 | 0,2435 | 0,3579 | 0,8121 | 0,4876 |

Random Forest hơn Decision Tree 1,90 điểm phần trăm AP và hơn Logistic Regression 3,40 điểm. Decision Tree của TV2 giới hạn độ sâu 10 và yêu cầu ít nhất 100 mẫu mỗi lá. Random Forest bình quân dự đoán từ 300 cây huấn luyện trên mẫu bootstrap và thử tập con đặc trưng ở mỗi nút. Trên test, Decision Tree cho 101 mức xác suất và Random Forest cho 8.147 mức, tạo thứ hạng chi tiết hơn. Cả hai dòng tinh chỉnh khớp bảng test của TV2.

Trên train, tương quan giữa `euribor3m`, `nr.employed` và `emp.var.rate` nằm trong khoảng 0,907–0,972. Logistic Regression bọc StandardScaler cho AP trung bình 5 fold là 0,4501 với đủ 64 biến và 0,4490 khi bỏ `euribor3m` và `nr.employed`. Notebook giữ đủ biến theo kết quả CV. Chênh lệch nhỏ cho thấy nhóm có thể cân nhắc bản gọn khi cần diễn giải hệ số hoặc giảm đầu vào.

## 5.2 Rò rỉ từ thời lượng cuộc gọi

Trên cùng các dòng test với tham số mô hình giữ nguyên, thêm `duration` làm AP tăng từ 0,4536 lên 0,6010 với Logistic Regression, từ 0,4686 lên 0,6342 với Decision Tree và từ 0,4876 lên 0,6657 với Random Forest. Mức tăng của Random Forest là 17,81 điểm phần trăm. Thời lượng cuộc gọi chỉ có sau khi cuộc gọi kết thúc, nên các kết quả có `duration` lượng hóa lợi thế từ thông tin tương lai. Kết quả dùng để chọn khách trước cuộc gọi lấy từ cấu hình bỏ cột này.

## 5.3 Đánh đổi khi cân bằng lớp

Với Random Forest tinh chỉnh và bỏ `duration`, dữ liệu gốc có AP trung bình 5 fold cao nhất (0,4692), tiếp theo là class weight (0,4685), oversampling (0,4677) và undersampling (0,4577). Chênh lệch giữa ba cấu hình đầu nhỏ so với độ lệch chuẩn giữa các fold. Oversampling đạt AP test 0,4889, sát mức 0,4876 của dữ liệu gốc.

Ở ngưỡng 0,5, oversampling có F1 test cao nhất (0,5017), với Recall 63,79% và Precision 41,34%. Số khách đăng ký được nhận diện tăng từ 226 lên 592, đồng thời dương tính giả tăng từ 109 lên 840. Class weight đạt F1 0,4916 và undersampling đạt 0,4944. Bảng này cho thấy đánh đổi giữa độ bao phủ và số cuộc gọi tại cùng ngưỡng. Kịch bản lợi nhuận của TV2 dùng cấu hình dữ liệu gốc, với ngưỡng riêng đã chọn trên train.

## 5.4 Đặc trưng và nhóm khách ưu tiên gọi

`nr.employed` có mức giảm độ hỗn tạp lớn nhất ở Decision Tree (0,6317). Random Forest phân bổ đóng góp nhiều hơn giữa `euribor3m` (0,1710), `nr.employed` (0,1539) và `poutcome` (0,0832 sau khi gộp các cột one-hot). Số điểm chia và tương quan giữa các biến vĩ mô có thể ảnh hưởng thứ hạng. Tỷ lệ đăng ký quan sát được giúp xác định nhóm ưu tiên gọi.

Nếu ngân sách chỉ đủ gọi khoảng 10% khách, có thể ưu tiên nhóm điểm Random Forest cao nhất khi bỏ `duration`. Trên test, nhóm này có 429 lượt đăng ký trong 824 khách (52,06%), so với 11,27% trên toàn tập. Trong 279 khách từng có chiến dịch thành công (`poutcome_success = 1`), 172 người đăng ký (61,65%). Lịch sử phản hồi tích cực là tín hiệu đã có trước cuộc gọi. Hai nhóm có thể trùng khách.

Với giả định chi phí 5 mỗi cuộc gọi và doanh thu 50 mỗi lượt đăng ký, TV2 chọn ngưỡng 0,10 trên dự đoán ngoài fold của train. Áp dụng ngưỡng này trên test cho 1.766 lượt gọi, 623 lượt đăng ký, Recall 67,13% và lợi nhuận mô phỏng 22.320 đơn vị. Ở ngưỡng 0,5, mô hình gọi 335 khách, nhận diện 226 lượt đăng ký và đạt 9.625 đơn vị. Kịch bản gọi toàn bộ 8.236 khách đạt 5.220 đơn vị theo cùng giả định.

## 5.5 Hiệu năng Spark và scikit-learn

Hai thư viện huấn luyện Decision Tree với độ sâu tối đa 8 và tối thiểu 20 mẫu ở nút lá. Dữ liệu là tập train bỏ `duration`, nhân bản theo 1×, 10×, 50× và 100× để mô phỏng tải. Thời gian tính trong `fit` sau khi dữ liệu đã chuẩn bị. RAM là RSS đỉnh lấy mẫu khoảng 20 ms, cộng bộ nhớ thường trú của Python và JVM ở phía Spark. Mỗi mốc đo ba lần sau một lượt làm nóng ở kích thước nhỏ nhất.

Phép đo thực hiện tuần tự trên Windows, Intel Core i5-12500H và 15,6 GiB RAM, với Python 3.12.13. Spark 4.1.4 chạy `local[1]` với heap 512 MiB và JDK 19.0.1. Decision Tree của scikit-learn 1.9.1 cũng fit bằng một luồng. Bảng ghi trung bình ba lần chạy.

| Mức | Dòng train | scikit-learn (giây) | Spark (giây) | scikit-learn (MiB) | Spark (MiB) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1× | 32.940 | 0,15 | 0,85 | 211,9 | 845,5 |
| 10× | 329.400 | 4,53 | 4,39 | 302,1 | 1.054,0 |
| 50× | 1.647.000 | 17,21 | 27,49 | 660,6 | 1.102,6 |
| 100× | 3.294.000 | 34,01 | 58,99 | 1.108,2 | 1.109,7 |

Ở mốc 1×, bộ nhớ nền của JVM và chi phí điều phối công việc chiếm phần đáng kể trong phép đo Spark. Ở 10×, chênh lệch thời gian khoảng 0,14 giây, nhỏ hơn độ lệch chuẩn của mỗi nhóm lần chạy (0,34 và 0,40 giây). Tại 50× và 100×, scikit-learn fit nhanh hơn. Ở mốc 100×, thời gian Spark gấp 1,73 lần và cả hai dùng xấp xỉ 1.110 MiB RSS đỉnh trung bình.

Khi dữ liệu vừa RAM và công việc chủ yếu là fit trên một máy, scikit-learn tránh được chi phí điều phối và JVM. Spark có lợi thế tiềm năng khi dữ liệu được phân vùng trên nhiều worker hoặc khi nhiều bước xử lý phải chạy song song trên tập dữ liệu vượt khả năng một máy. Cần đo trên cụm với cả thời gian đọc và chuẩn bị dữ liệu để đánh giá điều kiện đó.

## 5.6 Hạn chế

Tập dữ liệu sạch có 41.176 dòng. Chia ngẫu nhiên giúp giữ tỷ lệ lớp, đồng thời có thể đưa các quan sát thuộc cùng giai đoạn vào cả train và test. Khi huấn luyện lại Random Forest trên `train_time.parquet` và đánh giá trên `test_time.parquet`, ROC-AUC giảm từ 0,8121 xuống 0,7481. AP tăng từ 0,4876 lên 0,5459 và cần được đọc cùng tỷ lệ đăng ký nền tăng từ 11,27% lên 30,83%.

Ở ngưỡng 0,10 trên test theo thời gian, 5.037 lượt gọi nhận diện 2.181 khách đăng ký và đạt lợi nhuận mô phỏng 83.865 đơn vị. Gọi toàn bộ đạt 85.770 đơn vị trong giai đoạn này. Ngưỡng 0,13 chọn từ `train_time` cho kết quả 76.675 đơn vị trên test thời gian. Khác biệt giữa hai split cho thấy cần kiểm tra lại ngưỡng trên dữ liệu theo thời gian khi đánh giá chiến dịch mới.

Benchmark nhân bản các dòng hiện có để tăng tải, nên giữ nguyên phân bố và độ đa dạng của tập gốc. Hai thư viện cũng tìm điểm chia theo cách riêng, với Spark phân bin các giá trị liên tục. Bảng hiệu năng mô tả đúng cấu hình một máy đã đo.
