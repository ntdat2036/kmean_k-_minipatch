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
    KMeansPlusPlus,
    MiniBatchKMeans,
    PCA,
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    rand_index,
    Pipeline,
    map_cluster_profiles
)
from preprocess import build_features, validate_input_data, FEATURE_COLS

DATA_PATH = 'Wholesale_customers_data.csv'
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


def test_all_kmeans_variants():
    """Kiểm tra cả 3 biến thể K-Means (Standard, KMeans++, MiniBatchKMeans)"""
    np.random.seed(42)
    c1 = np.random.normal(loc=0.0, scale=0.5, size=(50, 2))
    c2 = np.random.normal(loc=5.0, scale=0.5, size=(50, 2))
    X = np.vstack([c1, c2])

    models = [
        KMeans(n_clusters=2, n_init=10, random_state=42),
        KMeansPlusPlus(n_clusters=2, n_init=10, random_state=42),
        MiniBatchKMeans(n_clusters=2, n_init=10, random_state=42)
    ]

    for model in models:
        labels = model.fit_predict(X)
        assert len(labels) == 100
        assert len(np.unique(labels)) == 2
        assert model.inertia_ > 0


def test_shared_preprocessing_and_validation():
    """Kiểm tra tiền xử lý tập trung và bắt lỗi dữ liệu vào không hợp lệ"""
    df = pd.read_csv(DATA_PATH)
    df_feat = build_features(df)
    assert df_feat.shape == (440, 9)

    # Test missing columns
    with pytest.raises(ValueError):
        build_features(df.drop(columns=['Fresh']))

    # Test negative values
    df_bad = df.copy()
    df_bad.loc[0, 'Fresh'] = -100
    with pytest.raises(ValueError):
        build_features(df_bad)


def test_dynamic_cluster_profiling():
    """Kiểm tra hàm ánh xạ cụm động map_cluster_profiles"""
    df = pd.read_csv(DATA_PATH)
    labels = np.array([0] * 150 + [1] * 150 + [2] * 140)
    profiles = map_cluster_profiles(df, labels)
    assert len(profiles) == 3
    assert any("VIP" in desc for desc in profiles.values())


def test_cluster_profiling_tie_break_conflict():
    """Kiem tra map_cluster_profiles khi 1 cum vua co Total_Spend cao nhat
    VUA co Fresh_Frozen_Ratio cao nhat (tinh huong tranh chap nhan VIP/HoReCa)."""
    df = pd.DataFrame({
        'Fresh':            [50000, 3000, 1000],
        'Milk':             [20000, 2000, 8000],
        'Grocery':          [20000, 1000, 9000],
        'Frozen':           [30000, 500,  500],
        'Detergents_Paper': [10000, 300,  4000],
        'Delicassen':       [5000,  200,  600],
    })
    # Cụm 0: Total_Spend cao NHẤT (135000) và Fresh_Frozen_Ratio cũng cao nhất (~0.59)
    labels = np.array([0, 1, 2])
    profiles = map_cluster_profiles(df, labels)

    assert profiles[0] != profiles[1] and profiles[1] != profiles[2] and profiles[0] != profiles[2], "3 cum phai co 3 nhan khac nhau"
    assert "VIP" in profiles[0] or "HoReCa" in profiles[0]
    assert len(set(profiles.values())) == 3, "Dam bao khong co 2 cum bi gan trung nhan"


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
    assert chi > 10.0, f"Calinski-Harabász Index phải lớn, thực tế: {chi}"
    assert dbi < 0.5, f"Davies-Bouldin Index phải nhỏ với cụm phân biệt, thực tế: {dbi}"


def test_rand_index_identical_labels():
    """Kiểm tra chỉ số Rand Index khi 2 bộ nhãn trùng khớp hoàn toàn"""
    labels = np.array([0, 0, 1, 1, 2, 2])
    assert rand_index(labels, labels) == 1.0


def test_rand_index_range():
    """Kiểm tra Rand Index với bộ nhãn hoán đổi thứ tự nhãn cụm (permutation invariant)"""
    a = np.array([0, 0, 1, 1])
    b = np.array([1, 1, 0, 0])
    assert rand_index(a, b) == 1.0


@pytest.mark.parametrize("model_cls", [KMeans, KMeansPlusPlus, MiniBatchKMeans])
def test_pipeline_and_inference_all_algorithms(model_cls):
    """Kiểm tra Pipeline và quy trình suy luận dự đoán trên cả 3 thuật toán"""
    df = pd.read_csv(DATA_PATH)
    X = build_features(df)

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('kmeans', model_cls(n_clusters=3, n_init=5, random_state=42))
    ])

    preds = pipeline.fit_predict(X.values)
    assert len(preds) == len(df)
    assert set(np.unique(preds)).issubset({0, 1, 2})

    new_preds = pipeline.predict(X.values[:5])
    assert len(new_preds) == 5


def test_preprocessed_csv():
    """Kiểm tra file kết quả preprocessed nếu tồn tại"""
    if os.path.exists(PREPROCESSED_PATH):
        df_prep = pd.read_csv(PREPROCESSED_PATH)
        assert 'Cluster' in df_prep.columns, "File wholesale_preprocessed.csv thiếu cột Cluster"
        assert list(df_prep.columns)[-1] == 'Cluster', "Cột Cluster phải nằm ở vị trí cuối cùng"


def test_predict_before_fit_raises_error():
    """Kiem tra goi predict() truoc fit() bao loi ro rang (RuntimeError)."""
    model = KMeans(n_clusters=3)
    X_dummy = np.random.rand(10, 4)
    with pytest.raises(RuntimeError):
        model.predict(X_dummy)


def test_n_iter_recorded_after_fit():
    """Kiem tra n_iter_ duoc ghi nhan sau khi fit (khong con None)."""
    X_dummy = np.random.rand(50, 3)
    model = KMeans(n_clusters=2, n_init=3, random_state=1)
    model.fit(X_dummy)
    assert model.n_iter_ is not None
    assert model.n_iter_ > 0


def test_near_zero_milk_ratio_clip():
    """Kiểm tra build_features không gây ra inf/nan và clip tỷ lệ khi Milk rất nhỏ hoặc bằng 0."""
    df = pd.read_csv(DATA_PATH).copy()
    df.loc[0, 'Milk'] = 0
    df_feat = build_features(df)
    assert not np.isnan(df_feat.values).any(), "Dữ liệu có chứa NaN sau build_features"
    assert not np.isinf(df_feat.values).any(), "Dữ liệu có chứa Inf sau build_features"


def test_out_of_sample_generalization():
    """Kiểm tra khả năng tổng quát hóa Out-of-Sample Train/Validation Split (80/20)"""
    df = pd.read_csv(DATA_PATH)
    X_raw = build_features(df).values

    np.random.seed(42)
    n_samples = len(X_raw)
    indices = np.random.permutation(n_samples)
    train_size = int(0.8 * n_samples)

    train_idx, val_idx = indices[:train_size], indices[train_size:]
    X_train_raw, X_val_raw = X_raw[train_idx], X_raw[val_idx]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)

    km = KMeansPlusPlus(n_clusters=3, n_init=10, random_state=42)
    train_labels = km.fit_predict(X_train)
    centroids = km.cluster_centers_

    val_dists = np.linalg.norm(X_val[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2)
    val_labels = np.argmin(val_dists, axis=1)

    train_sil = silhouette_score(X_train, train_labels)
    val_sil = silhouette_score(X_val, val_labels)

    # Đảm bảo Silhouette score trên tập Validation không sụt giảm quá 0.05 so với Train
    assert val_sil > 0.10, f"Validation Silhouette Score quá thấp: {val_sil}"
    assert abs(train_sil - val_sil) < 0.05, f"Chênh lệch Silhouette Train/Val quá lớn: {abs(train_sil - val_sil)}"






