# DỰ ÁN PHÂN KHÚC KHÁCH HÀNG BÁN BUÔN (WHOLESALE CUSTOMERS – K-MEANS CLUSTERING)

> **Lưu ý quan trọng**: Dự án này được xây dựng **100% bằng thuật toán tự cài đặt (Pure NumPy)**. **TUYỆT ĐỐI KHÔNG SỬ DỤNG THƯ VIỆN SCIKIT-LEARN (`sklearn`)** theo đúng yêu cầu học thuật.

---

## 1. TỔNG QUAN DỰ ÁN

Dự án thực hiện phân khúc 440 khách hàng của nhà phân phối bán buôn tại Bồ Đào Nha dựa trên hành vi chi tiêu hàng năm cho 6 nhóm mặt hàng:
- `Fresh` (Thực phẩm tươi sống)
- `Milk` (Sữa)
- `Grocery` (Hàng tạp hóa)
- `Frozen` (Thực phẩm đông lạnh)
- `Detergents_Paper` (Chất tẩy rửa & Giấy)
- `Delicassen` (Thực phẩm cao cấp)

### Các thành tựu chính:
1. **Tiền xử lý & Feature Engineering (`preprocess.py`)**: Đã xây dựng module tập trung `build_features` thực hiện biến đổi Logarithm (`log1p`) cho 6 ngành hàng gốc và 3 tỷ lệ chi tiêu (`Fresh_Ratio_log`, `NonEssential_Ratio_log`, `Grocery_Milk_Ratio_log`), bảo đảm 100% đồng bộ giữa train (`main.py`) và inference (`predict_new.py`).
2. **So sánh 3 biến thể K-Means (Lloyd -> K-Means++ -> Mini-Batch K-Means)**:
   - **K-Means (Lloyd)**: Thuật toán phân cụm cơ bản khởi tạo ngẫu nhiên.
   - **K-Means++**: Khởi tạo trọng số khoảng cách $D^2$ khắc phục nhạy cảm vị trí tâm ban đầu.
   - **Mini-Batch K-Means**: Cập nhật online theo lô nhỏ tối ưu tốc độ và bộ nhớ cho dữ liệu lớn.
   - So sánh thực nghiệm 3 phương pháp K-Means qua các cải tiến kỹ thuật, đánh giá bằng Silhouette/CHI/DBI trên cùng một bộ dữ liệu đã chuẩn hóa.
3. **Bộ chỉ số đánh giá đa chiều & Dynamic Selection**:
   - **Elbow Method & Silhouette Curve ($K = 2 \rightarrow 8$)**
   - **Composite Metric Ranking** tự động chọn mô hình xuất sắc nhất.
4. **Trực quan hóa PCA 2D**: Tự cài đặt thuật toán PCA bằng phân tích SVD với Sign Determinism để chiếu không gian 9D về 2D cố định trục.
5. **Đóng gói Pipeline & Dynamic Profiling**: Đóng gói `StandardScaler` + `KMeans` thành `Pipeline` và tự động ánh xạ nhãn cụm dựa trên chi tiêu thực tế (`predict_new.py`).

---

## 2. CẤU TRÚC THƯ MỤC DỰ ÁN

```text
Kmeans/
├── Wholesale customers data.csv     # Dữ liệu gốc (440 dòng x 8 cột)
├── Wholesale_Customers_KMeans.ipynb  # Jupyter Notebook báo cáo phân tích toàn diện
├── preprocess.py                     # Module tiền xử lý & Feature Engineering dùng chung duy nhất
├── model.py                          # Định nghĩa các mô hình & thuật toán (StandardScaler, KMeans, KMeansPlusPlus, MiniBatchKMeans, PCA, Metrics, Pipeline)
├── main.py                           # Script thực thi chính (Huấn luyện, so sánh 3 thuật toán & xuất mô hình)
├── predict_new.py                    # Script suy luận dự đoán phân cụm cho khách hàng mới
├── test_kmeans_pipeline.py           # Bộ kiểm thử tự động (pytest pass 100%)
├── requirements.txt                  # Khai báo phụ thuộc có pin version (pandas, numpy, matplotlib, pytest)
└── README.md                         # Báo cáo & tài liệu hướng dẫn
```

---

## 3. Ý NGHĨA KINH DOANH CỦA 3 CỤM KHÁCH HÀNG (K = 3)

- **Cụm 0 — Bán lẻ (Retail)**: Chi tiêu mạnh cho `Grocery`, `Milk` và `Detergents_Paper`. Phù hợp với các siêu thị nhỏ, cửa hàng tiện lợi.
- **Cụm 1 — Nhà hàng / Khách sạn (HoReCa)**: Chi tiêu áp đảo cho `Fresh` và `Frozen`. Phù hợp với các khách sạn, nhà hàng, quán ăn.
- **Cụm 2 — Khách hàng VIP / Cao cấp**: Mức chi tiêu lớn vượt trội trên tất cả 6 nhóm mặt hàng. Nhóm mang lại doanh thu cao nhất.

---

## 4. HƯỚNG DẪN CÀI ĐẶT VÀ VẬN HÀNH

### Bước 1: Cài đặt môi trường
Cài đặt các thư viện phụ thuộc (lưu ý: KHÔNG cài đặt `scikit-learn`):
```bash
pip install -r requirements.txt
```

### Bước 2: Chạy Jupyter Notebook
Mở và thực thi toàn bộ notebook báo cáo:
```bash
jupyter notebook Wholesale_Customers_KMeans.ipynb
```

### Bước 3: Chạy Script huấn luyện chính (`main.py`)
Thực thi toàn bộ quy trình từ nạp dữ liệu, huấn luyện mô hình đến xuất kết quả:
```bash
python main.py
```

### Bước 4: Chạy Unit Test kiểm thử tự động
Chạy bộ kiểm thử tự động với `pytest`:
```bash
pytest test_kmeans_pipeline.py
```

### Bước 5: Dự đoán dữ liệu mới (Inference)
Thực thi script suy luận dữ liệu khách hàng mới:
```bash
python predict_new.py
```

---

## 5. BỘ MÃ NGUỒN TỰ CÀI ĐẶT (`model.py`)

Thư viện [`model.py`](file:///d:/May_Hoc/Kmeans/model.py) bao gồm các lớp và hàm toán học tự định nghĩa:
- `StandardScaler`: Chuẩn hóa Z-score.
- `KMeans`: Phân cụm K-Means Lloyd + K-Means++ init + WCSS.
- `PCA`: Giảm chiều dữ liệu dựa trên SVD.
- `silhouette_score`: Tính Silhouette Coefficient cho từng mẫu.
- `calinski_harabasz_score`: Tính tỷ lệ phân tán giữa cụm và trong cụm.
- `davies_bouldin_score`: Tính chỉ số đo lường độ phân biệt giữa các cụm.
- `Pipeline`: Quản lý quy trình tiền xử lý và mô hình.
