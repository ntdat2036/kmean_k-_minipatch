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
1. **Tiền xử lý & Feature Engineering**: Biến đổi Logarithm (`log1p`), tính tỷ lệ tươi sống (`Fresh_Ratio`), tỷ lệ hàng không thiết yếu (`NonEssential_Ratio`), tỷ lệ tạp hóa/sữa (`Grocery_Milk_Ratio`) và tổng chi tiêu (`Total_Spend`).
2. **K-Means Clustering (NumPy)**: Tự cài đặt thuật toán K-Means với khởi tạo K-Means++, chạy đa khởi tạo (`n_init`), tính WCSS (Inertia) và gán cụm.
3. **Bộ chỉ số đánh giá đa chiều**:
   - **Elbow Method (Inertia / WCSS)**
   - **Silhouette Score**
   - **Calinski-Harabász Index (CHI)**
   - **Davies-Bouldin Index (DBI)**
4. **Trực quan hóa PCA 2D**: Tự cài đặt thuật toán PCA bằng phân tích SVD để chiếu không gian cao chiều về 2D.
5. **Đóng gói Pipeline & Inference**: Đóng gói `StandardScaler` + `KMeans` thành `Pipeline` để thực hiện suy luận trên dữ liệu mới (`predict_new.py`).

---

## 2. CẤU TRÚC THƯ MỤC DỰ ÁN

```text
Kmeans/
├── Wholesale customers data.csv     # Dữ liệu gốc (440 dòng x 8 cột)
├── wholesale_preprocessed.csv        # Dữ liệu đã xử lý kèm cột 'Cluster' làm cột cuối
├── Wholesale_Customers_KMeans.ipynb  # Jupyter Notebook báo cáo phân tích toàn diện
├── model.py                          # Định nghĩa các mô hình & thuật toán (StandardScaler, KMeans, PCA, Metrics, Pipeline)
├── main.py                           # Script thực thi chính (Huấn luyện, đánh giá & xuất mô hình)
├── predict_new.py                    # Script dự đoán phân cụm cho khách hàng mới
├── test_kmeans_pipeline.py           # Bộ kiểm thử tự động (pytest)
├── requirements.txt                  # Khai báo phụ thuộc (pandas, numpy, matplotlib, seaborn, pytest)
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
