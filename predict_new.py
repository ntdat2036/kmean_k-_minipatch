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
from model import (
    StandardScaler,
    KMeans,
    Pipeline,
    preprocess_features,
    map_cluster_profiles
)

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def load_pipeline(model_path='kmeans_pipeline.pkl'):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Không tìm thấy file mô hình: {model_path}. Vui lòng chạy main.py để huấn luyện và xuất mô hình.")
    with open(model_path, 'rb') as f:
        pipeline = pickle.load(f)
    return pipeline

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

    # Tiền xử lý tập trung (100% đồng bộ với main.py)
    X_new = preprocess_features(sample_customers)
    predictions = pipeline.predict(X_new)
    sample_customers['Cluster'] = predictions

    # Ánh xạ nhãn động dựa trên chi tiêu thực tế (Dynamic Profiling)
    profiles = map_cluster_profiles(sample_customers, predictions)

    print("\n" + "=" * 70)
    print("  KẾT QUẢ PHÂN KHÚC KHÁCH HÀNG MỚI (DYNAMIC CLUSTER MAPPING)")
    print("=" * 70)
    for idx, row in sample_customers.iterrows():
        c_id = int(row['Cluster'])
        desc = profiles.get(c_id, f"Cụm {c_id}")
        print(f"  Khách hàng {idx + 1:2d} | Cụm {c_id} => {desc}")
    print("=" * 70)

if __name__ == '__main__':
    main()
