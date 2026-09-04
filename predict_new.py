"""
Script suy luận (inference) phân nhóm khách hàng bán buôn cho dữ liệu mới
Sử dụng mô hình Pipeline K-Means đã huấn luyện (kmeans_pipeline.pkl)
Không phụ thuộc vào thư viện scikit-learn.
"""

import os
import sys
import pickle
import pandas as pd
import numpy as np
from model import StandardScaler, KMeans, Pipeline

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def load_pipeline(model_path='kmeans_pipeline.pkl'):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Không tìm thấy file mô hình: {model_path}. Vui lòng chạy notebook để huấn luyện và xuất mô hình.")
    with open(model_path, 'rb') as f:
        pipeline = pickle.load(f)
    return pipeline

def validate_input_data(df):
    """
    Kiểm tra tính hợp lệ của dữ liệu đầu vào trước khi thực hiện suy luận.
    - Kiểm tra đủ 6 cột đặc trưng chi tiêu
    - Kiểm tra kiểu dữ liệu số
    - Kiểm tra không chứa giá trị âm
    """
    required_cols = ['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Dữ liệu thiếu các cột chi tiêu bắt buộc: {missing}")

    for col in required_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise TypeError(f"Cột '{col}' phải là kiểu dữ liệu số.")

    if (df[required_cols] < 0).any().any():
        raise ValueError("Chi tiêu ngành hàng không được chứa giá trị âm.")

def preprocess_new_data(df):
    """
    Thực hiện Feature Engineering đồng bộ với main.py:
    1. Kiểm tra tính hợp lệ của dữ liệu vào (validate_input_data)
    2. Biến đổi log1p cho 6 nhóm chi tiêu: Fresh_log, Milk_log, ...
    3. Biến đổi log1p cho các tỷ lệ: Fresh_Ratio_log, NonEssential_Ratio_log, Grocery_Milk_Ratio_log
    """
    validate_input_data(df)
    df_proc = df.copy()
    feature_cols = ['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen']
    
    # 1. Log1p cho 6 nhóm chi tiêu
    for col in feature_cols:
        df_proc[f'{col}_log'] = np.log1p(df_proc[col])
        
    # 2. Log1p cho các tỷ lệ
    total_spend = df_proc[feature_cols].sum(axis=1)
    df_proc['Fresh_Ratio_log']        = np.log1p(df_proc['Fresh'] / (total_spend + 1e-6))
    df_proc['NonEssential_Ratio_log'] = np.log1p((df_proc['Grocery'] + df_proc['Detergents_Paper']) / (total_spend + 1e-6))
    df_proc['Grocery_Milk_Ratio_log'] = np.log1p(df_proc['Grocery'] / (df_proc['Milk'] + 1e-6))
    
    training_cols = [f'{col}_log' for col in feature_cols] + [
        'Fresh_Ratio_log', 'NonEssential_Ratio_log', 'Grocery_Milk_Ratio_log'
    ]
    return df_proc[training_cols]

def describe_clusters(df_result):
    cluster_descriptions = {
        0: "Cụm 0 — Bán lẻ phổ thông (Retail): Nhu cầu Tạp hóa, Sữa & Chất tẩy rửa cao",
        1: "Cụm 1 — Nhà hàng / Khách sạn (HoReCa): Nhu cầu Thực phẩm tươi sống & Đông lạnh cao",
        2: "Cụm 2 — Khách hàng VIP / Cao cấp: Tổng chi tiêu vượt trội ở tất cả các ngành hàng"
    }
    print("\n" + "="*70)
    print("  KẾT QUẢ PHÂN KHÚC KHÁCH HÀNG MỚI")
    print("="*70)
    for idx, row in df_result.iterrows():
        c_id = int(row['Cluster'])
        desc = cluster_descriptions.get(c_id, f"Cụm {c_id}")
        print(f"  Khách hàng {idx + 1:2d} | Cụm {c_id} => {desc}")
    print("="*70)

def main():
    print("Khởi tạo mô hình Pipeline K-Means...")
    try:
        pipeline = load_pipeline()
    except Exception as e:
        print(f"Lỗi: {e}")
        return

    # Dữ liệu mẫu 3 khách hàng thử nghiệm
    sample_customers = pd.DataFrame([
        # Khách hàng 1: Nhu cầu HoReCa tươi sống cao
        {'Channel': 1, 'Region': 3, 'Fresh': 35000, 'Milk': 2000, 'Grocery': 3000, 'Frozen': 9000, 'Detergents_Paper': 400, 'Delicassen': 1500},
        # Khách hàng 2: Nhu cầu Bán lẻ Retail cao
        {'Channel': 2, 'Region': 3, 'Fresh': 2000, 'Milk': 12000, 'Grocery': 18000, 'Frozen': 800, 'Detergents_Paper': 7000, 'Delicassen': 1200},
        # Khách hàng 3: Khách hàng VIP chi tiêu cực lớn
        {'Channel': 1, 'Region': 3, 'Fresh': 55000, 'Milk': 30000, 'Grocery': 45000, 'Frozen': 15000, 'Detergents_Paper': 18000, 'Delicassen': 10000}
    ])

    print("\n--- 3 Khách hàng thử nghiệm ---")
    print(sample_customers[['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen']])

    X_new = preprocess_new_data(sample_customers)
    predictions = pipeline.predict(X_new)
    sample_customers['Cluster'] = predictions

    describe_clusters(sample_customers)

if __name__ == '__main__':
    main()
