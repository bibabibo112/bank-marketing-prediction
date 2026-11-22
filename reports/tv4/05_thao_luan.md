# 5. Thảo luận

Các phép so sánh dùng cùng bộ Parquet của nhóm, gồm 32.940 dòng train và 8.236 dòng test. Có 928 khách đăng ký trong tập test (11,27%). PR-AUC trong phần này là Average Precision (AP) theo hàm đánh giá chung. Precision, Recall và F1 được tính ở ngưỡng 0,5.

## 5.1 Baseline và mô hình cây

Mô hình luôn dự đoán `no` đạt Accuracy 88,73% dù bỏ sót cả 928 khách đăng ký. Recall và F1 của lớp `yes` đều bằng 0, còn AP chỉ bằng tỷ lệ khách đăng ký (0,1127). Vì vậy AP và các chỉ số của lớp `yes` phù hợp hơn khi đánh giá khả năng lập danh sách gọi.

Với bộ dữ liệu bỏ `duration`, Logistic Regression đạt AP 0,4536, Random Forest đạt 0,4261 và Decision Tree đạt 0,1825. Random Forest có F1 cao hơn Logistic Regression ở ngưỡng 0,5 (0,3772 so với 0,3201), phản ánh sự khác nhau giữa xếp hạng xác suất và quyết định ở một ngưỡng. Hai mô hình cây ở đây dùng cấu hình mặc định để đối chứng. Kết luận về mô hình cây tinh chỉnh cần đối chiếu với bảng của TV2 trên cùng split.

Random Forest hơn Decision Tree 24,36 điểm phần trăm AP và đạt ROC-AUC 0,7772 so với 0,6254. Cây đơn mặc định chỉ tạo 6 mức xác suất trên test, nên nhiều khách nhận cùng điểm. Random Forest lấy trung bình dự đoán từ nhiều cây huấn luyện trên các mẫu bootstrap và tập con đặc trưng. Trong lần chạy này, mô hình tạo 676 mức xác suất, cho thứ hạng chi tiết hơn và giảm phụ thuộc vào các điểm chia của một cây.

Trên train, tương quan giữa `euribor3m`, `nr.employed` và `emp.var.rate` nằm trong khoảng 0,907–0,972. Logistic Regression bọc StandardScaler cho AP trung bình 5 fold là 0,4501 với đủ 64 biến và 0,4490 khi bỏ `euribor3m` và `nr.employed`. Notebook giữ đủ biến theo kết quả CV. Chênh lệch nhỏ cho thấy nhóm có thể cân nhắc bản gọn khi cần diễn giải hệ số hoặc giảm đầu vào.

## 5.2 Rò rỉ từ thời lượng cuộc gọi

Trên cùng các dòng test với tham số mô hình giữ nguyên, thêm `duration` làm AP tăng từ 0,4536 lên 0,6010 với Logistic Regression, từ 0,1825 lên 0,3165 với Decision Tree và từ 0,4261 lên 0,6536 với Random Forest. Mức tăng của Random Forest là 22,75 điểm phần trăm. Thời lượng cuộc gọi chỉ có sau khi cuộc gọi kết thúc, nên các kết quả có `duration` lượng hóa lợi thế từ thông tin tương lai. Kết quả dùng để chọn khách trước cuộc gọi lấy từ cấu hình bỏ cột này.

## 5.3 Đánh đổi khi cân bằng lớp

Với Random Forest bỏ `duration`, undersampling có AP trung bình trên train cao nhất (0,4132 so với 0,4117 của cấu hình gốc). Chênh lệch 0,0014 nhỏ so với độ lệch chuẩn giữa các fold (0,0206 và 0,0068). Trên test, undersampling phát hiện 632 khách đăng ký, tăng từ 268, đồng thời tăng dương tính giả từ 225 lên 1.440. Recall đạt 68,10% và Precision còn 30,50%.

Class weight đạt F1 test cao nhất (0,4518), với Recall 45,69% và 525 dương tính giả. Oversampling đạt F1 0,4345. Số liệu ở ngưỡng 0,5 cho thấy các lựa chọn đánh đổi số khách phát hiện được với khối lượng gọi. Để chọn cấu hình theo lợi nhuận, nhóm cần áp dụng cùng quy tắc chi phí và ngưỡng chọn trên validation của TV2.

## 5.4 Đặc trưng và nhóm khách ưu tiên gọi

`age` có mức giảm độ hỗn tạp lớn nhất ở cả Decision Tree (0,1618) và Random Forest (0,1480) sau khi gộp các cột one-hot theo biến gốc. `nr.employed` đứng thứ hai ở Decision Tree, còn `euribor3m` đứng thứ hai ở Random Forest. `job` chiếm khoảng 0,10 ở cả hai mô hình. Số điểm chia và tương quan giữa biến vĩ mô có thể ảnh hưởng thứ hạng này. Để xác định nhóm ưu tiên, cần xem hướng dự đoán và tỷ lệ đăng ký quan sát được.

Nếu ngân sách chỉ đủ gọi khoảng 10% khách, có thể ưu tiên nhóm điểm Random Forest cao nhất khi bỏ `duration`. Trên test, nhóm này có 396 lượt đăng ký trong 824 khách (48,06%), so với 11,27% trên toàn tập. Trong 279 khách từng có chiến dịch thành công (`poutcome_success = 1`), 172 người đăng ký (61,65%). Lịch sử phản hồi tích cực là tín hiệu đã có trước cuộc gọi. Hai nhóm có thể trùng khách, nên ngân sách và ngưỡng cuối cùng cần được chốt cùng TV2 theo số lượt gọi và chi phí.

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

Tập dữ liệu sạch có 41.176 dòng và lớp đăng ký chỉ chiếm 11,27% trên test. Chia ngẫu nhiên giúp giữ tỷ lệ lớp, đồng thời có thể đưa các quan sát thuộc cùng giai đoạn vào cả train và test. Khi đánh giá chiến dịch tương lai, nên huấn luyện trên `train_time.parquet` và kiểm tra trên `test_time.parquet` để đo ảnh hưởng của biến động theo thời gian.

Benchmark nhân bản các dòng hiện có để tăng tải, nên giữ nguyên phân bố và độ đa dạng của tập gốc. Hai thư viện cũng tìm điểm chia theo cách riêng, với Spark phân bin các giá trị liên tục. Bảng hiệu năng mô tả đúng cấu hình một máy đã đo.
