# DỰ ÁN PHÂN KHÚC KHÁCH HÀNG BÁN BUÔN (WHOLESALE CUSTOMERS – K-MEANS CLUSTERING)

> **Cam kết kỹ thuật**: Dự án được xây dựng **100% bằng thuật toán tự cài đặt (Pure NumPy)**. **TUYỆT ĐỐI KHÔNG SỬ DỤNG THƯ VIỆN SCIKIT-LEARN (`sklearn`)** theo đúng chuẩn mực học thuật bài bản.

---

## 1. TỔNG QUAN DỰ ÁN

Dự án thực hiện phân khúc **440 khách hàng** của nhà phân phối bán buôn tại Bồ Đào Nha dựa trên hành vi chi tiêu hàng năm cho 6 nhóm mặt hàng gốc:
- `Fresh`: Thực phẩm tươi sống
- `Milk`: Sữa và các sản phẩm từ sữa
- `Grocery`: Hàng tạp hóa
- `Frozen`: Thực phẩm đông lạnh
- `Detergents_Paper`: Chất tẩy rửa & Giấy
- `Delicassen`: Thực phẩm chế biến / Cao cấp

### Các Thành Tựu & Điểm Cải Tiến Kỹ Thuật Nổi Bật

1. **Tiền Xử Lý & Feature Engineering Dùng Chung (`preprocess.py`)**:
   - Xây dựng module tập trung `build_features` duy nhất cho toàn bộ project.
   - Áp dụng biến đổi Logarithm ($\log(1+x)$) cho 6 ngành hàng gốc nhằm giảm độ lệch (skewness).
   - Tạo thêm 3 tỷ lệ chi tiêu đặc trưng dựa trên tổng chi tiêu `total_spend`: `Fresh_Ratio_log`, `NonEssential_Ratio_log`, `Grocery_Milk_Ratio_log` (biến `total_spend` chỉ làm mẫu số trung gian, không đưa trực tiếp vào không gian đặc trưng huấn luyện).
   - Bảo đảm **100% nhất quán đặc trưng** giữa quá trình Huấn luyện (`main.py`) và Suy luận dữ liệu mới (`predict_new.py`).

2. **So Sánh 3 Biến Thể K-Means Công Bằng ($n\_init = 10$)**:
   - **K-Means (Lloyd Standard)**: Phân cụm truyền thống khởi tạo tâm ngẫu nhiên.
   - **K-Means++**: Khởi tạo vị trí tâm ban đầu dựa trên phân phối xác suất khoảng cách $D^2$ kết hợp Oversampling ($k \cdot m$), khắc phục nhược điểm nhạy cảm vị trí tâm.
   - **MiniBatchKMeans**: Cập nhật tâm theo từng lô nhỏ (Mini-batch) giúp tăng tốc độ huấn luyện và tối ưu bộ nhớ.
   - Đồng nhất $n\_init = 10$ cho cả 3 thuật toán để so sánh hiệu năng và thời gian công bằng.

3. **Cố Định Số Cụm $K = 3$ Theo Yêu Cầu Bài Toán Nghiệp Vụ (`BEST_K = 3`)**:
   - Hệ thống cố định `BEST_K = 3` nhất quán xuyên suốt quy trình huấn luyện, trực quan hóa và lập hồ sơ.
   - Hàm `find_optimal_k` thực hiện khảo sát tham khảo $K \in [2, 8]$ bằng `KMeansPlusPlus` để đối chứng chỉ số Elbow/Silhouette định lượng, in thông báo rõ ràng về lý do thiết lập cố định $K=3$ phù hợp với cấu trúc phân khúc doanh nghiệp.

4. **Tự Động Chọn Mô Hình Xuất Sắc Nhất (Dynamic Model Selection)**:
   - Đánh giá đa chiều qua 3 chỉ số: **Silhouette Score** (cao -> tốt), **Calinski-Harabasz Index (CHI)** (cao -> tốt), **Davies-Bouldin Index (DBI)** (thấp -> tốt).
   - Chuẩn hóa Min-Max liên tục trên cùng thang đo $[0, 1]$ để tính điểm Tổng hợp (Composite Metric Score):
     $$\text{Score} = \text{Norm}(\text{Silhouette}) + \text{Norm}(\text{CHI}) + \text{Norm}(1 - \text{DBI})$$
   - Loại bỏ rủi ro thắng nhị phân do chênh lệch nhỏ như nhiễu số thực.

5. **Lập Hồ Sơ Cụm Động Chuẩn Hóa Z-Score (Dynamic Cluster Profiling)**:
   - Hàm `map_cluster_profiles` sử dụng chuẩn hóa Z-score đồng thời trên 2 tiêu chí (`Total_Spend` và `Fresh_Frozen_Ratio`) để gán nhãn ưu tiên theo độ tách biệt vượt trội rõ nhất trước (xử lý tình huống tranh chấp nhãn giữa VIP và HoReCa).
   - Xuất file cấu hình ánh xạ tĩnh `cluster_profiles.json` làm tham chiếu nhất quán 100% cho bước suy luận dữ liệu mới (`predict_new.py`).

6. **Giảm Chiều PCA 2D & Trực Quan Hóa**:
   - Tự cài đặt **PCA bằng phân tích SVD** kết hợp chuẩn hóa dấu định hướng (Sign Determinism) để chiếu không gian 9D về 2D cố định trục.
   - Trực quan hóa so sánh cụm PCA 2D và đồ thị hội tụ của MiniBatchKMeans.

7. **Đóng Gói Pipeline & Bộ Kiểm Thử Tự Động (Unit Test)**:
   - Đóng gói `StandardScaler` và mô hình phân cụm vào `Pipeline` tự viết, lưu ra file `kmeans_pipeline.pkl`.
   - Bộ kiểm thử tự động với `pytest` kiểm tra 100% tính toàn vẹn dữ liệu, kiểm thử parameterized trên cả 3 thuật toán phân cụm, kiểm định tính ổn định `rand_index` và xử lý trường hợp tranh chấp nhãn.

---

## 2. CẤU TRÚC THƯ MỤC DỰ ÁN

```text
Kmeans/
├── Wholesale customers data.csv    # Dữ liệu gốc (440 mẫu x 8 cột)
├── preprocess.py                    # Module tiền xử lý & Feature Engineering dùng chung duy nhất
├── model.py                         # Định nghĩa toàn bộ thuật toán Pure NumPy (Scaler, KMeans, KMeans++, MiniBatch, PCA, Metrics, Pipeline, map_cluster_profiles, rand_index)
├── main.py                          # Script huấn luyện chính (Hằng số BEST_K=3, so sánh 3 thuật toán, xuất Pipeline & CSV)
├── predict_new.py                   # Script suy luận dự đoán phân cụm cho dữ liệu khách hàng mới (dùng nhãn cố định từ train)
├── Wholesale_Customers_KMeans.ipynb # Jupyter Notebook báo cáo phân tích & thực thi tương tác đồ án
├── cluster_profiles.json            # File nạp hồ sơ nhãn kinh doanh phân cụm cố định từ tập train
├── test_kmeans_pipeline.py          # Bộ kiểm thử tự động với Pytest (kiểm thử 3 thuật toán, tie-break, rand_index)
├── requirements.txt                 # Khai báo các thư viện phụ thuộc (NumPy, Pandas, Matplotlib, Seaborn, SciPy, Pytest)
└── README.md                        # Tài liệu hướng dẫn & Báo cáo kỹ thuật dự án
```

---

## 3. Ý NGHĨA KINH DOANH CỦA 3 CỤM KHÁCH HÀNG ($K = 3$)

- **Cụm VIP / Cao cấp**: Khách hàng có tổng chi tiêu lớn vượt trội trên toàn bộ 6 ngành hàng. Nhóm mang lại doanh thu chính, cần có chính sách chăm sóc đặc biệt và chiết khấu cao.
- **Cụm Nhà hàng / Khách sạn (HoReCa)**: Khách hàng có tỷ trọng chi tiêu áp đảo cho `Fresh` (Thực phẩm tươi) và `Frozen` (Thực phẩm đông lạnh). Phù hợp cho các chiến dịch tiếp thị thực phẩm chế biến tươi sống.
- **Cụm Bán lẻ Phổ thông (Retail)**: Khách hàng tập trung chi tiêu chính vào `Grocery` (Tạp hóa), `Milk` (Sữa) và `Detergents_Paper` (Chất tẩy rửa). Phù hợp cho các chương trình khuyến mãi hàng tiêu dùng nhanh (FMCG).

---

## 4. HƯỚNG DẪN CÀI ĐẶT VÀ VẬN HÀNH

### Bước 1: Cài đặt môi trường
Cài đặt các thư viện phụ thuộc bắt buộc (*lưu ý: Tuyệt đối không cài đặt hay sử dụng `scikit-learn`*):
```bash
pip install -r requirements.txt
```

### Bước 2: Chạy Script huấn luyện & so sánh chính (`main.py`)
Thực thi toàn bộ quy trình từ nạp dữ liệu, chuẩn hóa, phân tích $K$ tối ưu, huấn luyện 3 thuật toán đến xuất file kết quả:
```bash
python main.py
```
*Kết quả xuất ra bao gồm:*
- `wholesale_preprocessed.csv`: File dữ liệu đã gắn nhãn cụm `Cluster`.
- `kmeans_pipeline.pkl`: Mô hình Pipeline đã được đóng gói.
- `cluster_profiles.json`: Cấu hình ánh xạ nhãn kinh doanh cố định cho bước suy luận.
- Các đồ thị trực quan hóa: `elbow_silhouette_k.png`, `comparison_pca.png`, `convergence_minibatch.png`.

### Bước 3: Chạy Unit Test kiểm thử tự động
Thực thi bộ test tự động để đảm bảo tính đúng đắn của toàn bộ thuật toán:
```bash
pytest
```

### Bước 4: Suy luận dự đoán dữ liệu mới (`predict_new.py`)
Thực thi suy luận phân cụm cho các mẫu khách hàng mới:
```bash
python predict_new.py
```

---

## 5. THƯ VIỆN THUẬT TOÁN TỰ CÀI ĐẶT (`model.py`)

Toàn bộ các thuật toán trong [model.py](file:///d:/May_Hoc/Kmeans/model.py) được cài đặt hoàn toàn dựa trên các phép toán đại số tuyến tính của **NumPy**:

| Lớp / Hàm | Mô tả toán học & Thuật toán |
| :--- | :--- |
| `StandardScaler` | Chuẩn hóa Z-score: $z = \frac{x - \mu}{\sigma}$ (mean = 0, std = 1). |
| `KMeans` | Thuật toán Lloyd chuẩn ngẫu nhiên: Fit-predict, cập nhật ma trận khoảng cách Euclidean và tâm cụm. |
| `KMeansPlusPlus` | Khởi tạo tâm $D^2$-weighted sampling + Oversampling ($k \cdot m$), giảm tối đa WCSS ban đầu. |
| `MiniBatchKMeans` | Cập nhật tâm online bằng quy hoạch động giảm nhẹ độ phức tạp tính toán: $c_j \leftarrow (1-\eta)c_j + \eta x_i$. |
| `PCA` | Phân tích giá trị đơn lẻ SVD ($\mathbf{X} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T$), kết hợp chuẩn hóa dấu định hướng (Sign Determinism). |
| `silhouette_score` | Tính khoảng cách trung bình nội cụm $a(i)$ và khoảng cách lân cận nhỏ nhất $b(i)$: $s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$. |
| `calinski_harabasz_score` | Tỷ lệ tổng độ phân tán giữa các cụm (SSB) và trong nội cụm (SSW). |
| `davies_bouldin_score` | Đo lường độ tương đồng giữa cụm $i$ và cụm $j$ dựa trên bán kính phân tán $S_i, S_j$ và khoảng cách tâm $R_{ij}$. |
| `rand_index` | Tính chỉ số Rand Index đánh giá sự tương đồng giữa 2 kết quả phân cụm bất kỳ. |
| `Pipeline` | Đóng gói biến đổi dữ liệu (`StandardScaler`) và mô hình dự đoán thành một đối tượng duy nhất. |
| `map_cluster_profiles` | Phân tích Z-score đặc trưng chi tiêu thực tế để gán nhãn mô tả doanh nghiệp tự động & xử lý tranh chấp nhãn. |
