# -*- coding: utf-8 -*-
"""
weights.py
==========
    - scaler weights : mean_, scale_ của StandardScaler
    - model weights   : cluster_centers_ (tâm cụm) của cả 3 thuật toán
                         KMeans / KMeans++ / MiniBatchKMeans
    - D^2-weighting   : minh họa cơ chế trọng số D^2 dùng trong bước khởi tạo
                         của KMeans++ / MiniBatchKMeans

Chạy:  python weights.py
"""

import sys
import json
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from model import StandardScaler, KMeans, KMeansPlusPlus, MiniBatchKMeans
from preprocess import build_features, FEATURE_COLS

# Hyperparameter tối ưu (đồng bộ với main.py / notebook) — chỉ dùng để train,
# KHÔNG ghi vào file JSON output.
BEST_K = 3
RANDOM_STATE = 42
N_INIT = 10
KPP_OVERSAMPLE_FACTOR = 3
MB_BATCH_SIZE = 100
MB_MAX_NO_IMPROVEMENT = 15

OUTPUT_JSON = "model_weights.json"


def d2_weighting_demo(X, rng, n_show=5):
    """Minh họa D^2-weighting: P(x_i) = D(x_i)^2 / sum_j D(x_j)^2."""
    first_idx = rng.randint(0, X.shape[0])
    center0 = X[first_idx]
    dist_sq = np.sum((X - center0) ** 2, axis=1)
    probs = dist_sq / np.sum(dist_sq)
    top_idx = np.argsort(probs)[::-1][:n_show]
    return {
        "explanation": (
            "P(x_i) = D(x_i)^2 / sum_j D(x_j)^2 — diem cang XA tam da chon thi "
            "xac suat duoc chon lam tam KE TIEP cang cao (D^2-weighted sampling)."
        ),
        "first_center_index": int(first_idx),
        "top_candidate_points": [
            {"row_index": int(i), "distance_sq": float(dist_sq[i]), "probability": float(probs[i])}
            for i in top_idx
        ],
    }


def print_section(title):
    print("\n" + "=" * 78)
    print(f"  {title}")
    print("=" * 78)


def main():
    print_section("BUOC 1: NAP & TIEN XU LY DU LIEU")
    df_raw = pd.read_csv("Wholesale_customers_data.csv")
    df_feat = build_features(df_raw)
    feature_names = list(df_feat.columns)
    X = df_feat.values
    print(f"  So mau: {X.shape[0]} | So dac trung: {X.shape[1]}")
    print(f"  Dac trung: {feature_names}")

    print_section("BUOC 2: SCALER WEIGHTS (StandardScaler)")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    for name, mean, scale in zip(feature_names, scaler.mean_, scaler.scale_):
        print(f"  {name:<28} mean_ = {mean:>10.4f}   scale_ = {scale:>10.4f}")

    print_section("BUOC 3: HUAN LUYEN & TRICH XUAT MODEL WEIGHTS (cluster_centers_)")
    models = {
        "KMeans (Random Init)": KMeans(
            n_clusters=BEST_K, n_init=N_INIT, random_state=RANDOM_STATE
        ),
        "KMeansPlusPlus": KMeansPlusPlus(
            n_clusters=BEST_K, n_init=N_INIT, random_state=RANDOM_STATE,
            oversample_factor=KPP_OVERSAMPLE_FACTOR
        ),
        "MiniBatchKMeans": MiniBatchKMeans(
            n_clusters=BEST_K, n_init=N_INIT, random_state=RANDOM_STATE,
            batch_size=MB_BATCH_SIZE, max_no_improvement=MB_MAX_NO_IMPROVEMENT
        ),
    }

    model_weights = {}
    for name, model in models.items():
        model.fit(X_scaled)
        centers_original = scaler.inverse_transform(model.cluster_centers_)
        model_weights[name] = {
            "n_clusters": BEST_K,
            "inertia_": float(model.inertia_),
            "n_iter_": int(model.n_iter_),
            "cluster_centers_scaled": model.cluster_centers_.round(6).tolist(),
            "cluster_centers_original_scale": centers_original.round(2).tolist(),
        }

        print(f"\n  --- {name} ---")
        print(f"    inertia_ = {model.inertia_:.4f} | n_iter_ = {model.n_iter_}")
        print(f"    cluster_centers_ (da chuan hoa, shape {model.cluster_centers_.shape}):")
        for k, center in enumerate(model.cluster_centers_):
            vals = ", ".join(f"{v:.3f}" for v in center)
            print(f"      Cum {k}: [{vals}]")

    print_section("BUOC 4: MINH HOA D^2-WEIGHTING (khoi tao KMeans++ / MiniBatchKMeans)")
    rng_demo = np.random.RandomState(RANDOM_STATE)
    d2_demo = d2_weighting_demo(X_scaled, rng_demo)
    print(f"  {d2_demo['explanation']}")
    print(f"  Tam dau tien duoc chon ngau nhien: mau thu #{d2_demo['first_center_index']}")
    print(f"  Top {len(d2_demo['top_candidate_points'])} diem co xac suat duoc chon lam tam ke tiep CAO NHAT:")
    for p in d2_demo["top_candidate_points"]:
        print(f"    Mau #{p['row_index']:<5} D^2 = {p['distance_sq']:>10.4f}   P = {p['probability']:.6f}")

    print_section("BUOC 5: LUU KET QUA RA FILE JSON")
    output = {
        "note": (
            "K-Means khong co 'weight' kieu neural network. 'Trong so' hoc duoc "
            "cua mo hinh la TAM CUM (cluster_centers_) va tham so chuan hoa "
            "(scaler mean_/scale_). Hyperparameter da duoc khai bao trong main.py/notebook."
        ),
        "feature_names": feature_names,
        "scaler_weights": {
            "mean_": scaler.mean_.round(6).tolist(),
            "scale_": scaler.scale_.round(6).tolist(),
        },
        "model_weights_per_algorithm": model_weights,
        "kmeans_pp_d2_weighting_demo": d2_demo,
    }
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"  Da luu: {OUTPUT_JSON}")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()
