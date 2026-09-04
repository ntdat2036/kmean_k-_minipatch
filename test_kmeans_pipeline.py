"""
Bộ kiểm thử tự động (Unit Test) cho dự án Wholesale Customers K-Means.
Sử dụng pytest, hoàn toàn không phụ thuộc vào scikit-learn.
"""

import os
import pickle
import pandas as pd
import numpy as np
import pytest

from model import (
    StandardScaler,
    KMeans,
    PCA,
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    Pipeline
)

DATA_PATH = 'Wholesale customers data.csv'
MODEL_PATH = 'kmeans_pipeline.pkl'
PREPROCESSED_PATH = 'wholesale_preprocessed.csv'


def test_data_integrity():
    """Kiểm tra tính toàn vẹn của file dữ liệu đầu vào gốc"""
    assert os.path.exists(DATA_PATH), f"Không tìm thấy file dữ liệu gốc {DATA_PATH}"
    df = pd.read_csv(DATA_PATH)
    assert len(df) == 440, f"Kích thước số dòng không đúng, kỳ vọng 440, thực tế {len(df)}"
    expected_cols = ['Channel', 'Region', 'Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen']
    assert list(df.columns) == expected_cols, f"Các cột gốc không đúng cấu trúc"
    assert df.isnull().sum().sum() == 0, "Dữ liệu chứa giá trị null/missing"


def test_custom_standard_scaler():
    """Kiểm tra bộ chuẩn hóa StandardScaler tự viết bằng NumPy"""
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    np.testing.assert_allclose(np.mean(X_scaled, axis=0), [0.0, 0.0], atol=1e-7)
    np.testing.assert_allclose(np.std(X_scaled, axis=0), [1.0, 1.0], atol=1e-7)


def test_custom_kmeans():
    """Kiểm tra thuật toán K-Means phân cụm tự viết bằng NumPy"""
    np.random.seed(42)
    c1 = np.random.normal(loc=0.0, scale=0.5, size=(50, 2))
    c2 = np.random.normal(loc=5.0, scale=0.5, size=(50, 2))
    X = np.vstack([c1, c2])
    
    km = KMeans(n_clusters=2, n_init=10, random_state=42)
    labels = km.fit_predict(X)
    
    assert len(labels) == 100
    assert len(np.unique(labels)) == 2
    assert km.inertia_ > 0


def test_custom_pca():
    """Kiểm tra thuật toán PCA giảm chiều dữ liệu tự viết bằng NumPy"""
    X = np.random.normal(size=(100, 5))
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    assert X_pca.shape == (100, 2)
    assert len(pca.explained_variance_ratio_) == 2
    assert np.sum(pca.explained_variance_ratio_) <= 1.0


def test_metrics():
    """Kiểm tra các hàm tính chỉ số đánh giá phân cụm"""
    X = np.array([[0.0, 0.0], [0.1, 0.1], [10.0, 10.0], [10.1, 10.1]])
    labels = np.array([0, 0, 1, 1])
    
    sil = silhouette_score(X, labels)
    chi = calinski_harabasz_score(X, labels)
    dbi = davies_bouldin_score(X, labels)
    
    assert sil > 0.8, f"Silhouette Score phải cao với cụm rõ ràng, thực tế: {sil}"
    assert chi > 10.0, f"Calinski-Harabasz Index phải lớn, thực tế: {chi}"
    assert dbi < 0.5, f"Davies-Bouldin Index phải nhỏ với cụm phân biệt, thực tế: {dbi}"


def test_pipeline_and_inference():
    """Kiểm tra Pipeline và quy trình suy luận dự đoán"""
    df = pd.read_csv(DATA_PATH)
    spend_cols = ['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen']
    X = np.log1p(df[spend_cols].values)
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('kmeans', KMeans(n_clusters=3, n_init=5, random_state=42))
    ])
    
    preds = pipeline.fit_predict(X)
    assert len(preds) == len(df)
    assert set(np.unique(preds)).issubset({0, 1, 2})


def test_preprocessed_csv():
    """Kiểm tra file kết quả preprocessed nếu tồn tại"""
    if os.path.exists(PREPROCESSED_PATH):
        df_prep = pd.read_csv(PREPROCESSED_PATH)
        assert 'Cluster' in df_prep.columns, "File wholesale_preprocessed.csv thiếu cột Cluster"
        assert list(df_prep.columns)[-1] == 'Cluster', "Cột Cluster phải nằm ở vị trí cuối cùng"
