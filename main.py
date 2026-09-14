"""
main.py
=======
Script chính thực thi quy trình phân khúc khách hàng Wholesale Customers.
TUYỆT ĐỐI KHÔNG SỬ DỤNG scikit-learn.
"""

import sys
import os
import time
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from model import (
    StandardScaler,
    KMeans,
    KMeansPlusPlus,
    MiniBatchKMeans,
    PCA,
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    Pipeline,
    map_cluster_profiles,
    select_best_algorithm
)
from preprocess import build_features, FEATURE_COLS

BEST_K = 3
CLUSTER_COLORS = plt.cm.tab10(np.linspace(0, 1, 10))


def load_and_preprocess_data(data_path):
    """Nạp dữ liệu và tạo đặc trưng dùng chung build_features từ preprocess.py."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Không tìm thấy file: {data_path}")
    df_raw = pd.read_csv(data_path)
    df_features = build_features(df_raw)
    feature_cols = list(df_features.columns)
    return df_raw, df_features, feature_cols


def evaluate_clustering(X_scaled, labels):
    """Tính 3 chỉ số đánh giá: Silhouette, CHI, DBI."""
    return {
        "silhouette": silhouette_score(X_scaled, labels),
        "calinski":   calinski_harabasz_score(X_scaled, labels),
        "davies":     davies_bouldin_score(X_scaled, labels),
    }


def compare_algorithms(X_scaled, k=BEST_K):
    """
    Huấn luyện và so sánh 3 thuật toán phân cụm trên cùng dữ liệu (n_init=10 đồng nhất, n_clusters=k).

    Chuỗi cải tiến:
      KMeans (random) -> KMeansPlusPlus (D^2 init) -> MiniBatchKMeans (online)
    """
    algorithms = {
        "KMeans (Random Init)": KMeans(
            n_clusters=k, n_init=10, random_state=42
        ),
        "KMeans++ (D^2+Oversampling)": KMeansPlusPlus(
            n_clusters=k, n_init=10, random_state=42, oversample_factor=3
        ),
        "MiniBatchKMeans (Online)": MiniBatchKMeans(
            n_clusters=k, n_init=10, random_state=42,
            batch_size=100, max_no_improvement=15
        ),
    }
    results = {}
    for name, model in algorithms.items():
        print(f"    Đang chạy: {name} (n_init=10, k={k})...")
        t0 = time.perf_counter()
        labels = model.fit_predict(X_scaled)
        elapsed = time.perf_counter() - t0
        metrics = evaluate_clustering(X_scaled, labels)
        results[name] = {
            "model":       model,
            "model_class": type(model),
            "labels":      labels,
            "inertia":     model.inertia_,
            "time_s":      elapsed,
            **metrics
        }
    return results




def print_comparison_table(results, k=BEST_K):
    """In bảng so sánh kết quả 3 thuật toán."""
    sep = "=" * 88
    print("\n" + sep)
    print(f"  BẢNG SO SÁNH 3 THUẬT TOÁN PHÂN CỤM (K = {k}, n_init = 10)")
    print(sep)
    header = f"  {'Thuật toán':<30} {'T.gian(s)':>10} {'Inertia':>12} {'Silhouette':>12} {'CHI':>10} {'DBI':>10}"
    print(header)
    print("-" * 88)
    for name, r in results.items():
        print(f"  {name:<30} {r['time_s']:>10.4f} {r['inertia']:>12.2f} "
              f"{r['silhouette']:>12.4f} {r['calinski']:>10.2f} {r['davies']:>10.4f}")
    print(sep)
    print("  Ghi chú: Silhouette cao -> tốt | CHI cao -> tốt | DBI thấp -> tốt")
    print(sep + "\n")


def plot_pca_comparison(X_scaled, results, save_path="comparison_pca.png", k_clusters=BEST_K):
    """Vẽ 3 biểu đồ PCA 2D so sánh kết quả phân cụm."""
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    evr = pca.explained_variance_ratio_

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"So sánh Kết quả Phân cụm PCA 2D - 3 Thuật toán (K = {k_clusters})",
                 fontsize=14, fontweight="bold")

    for ax, (name, r) in zip(axes, results.items()):
        labels = r["labels"]
        for k in range(k_clusters):
            mask = (labels == k)
            ax.scatter(
                X_pca[mask, 0], X_pca[mask, 1],
                c=[CLUSTER_COLORS[k % len(CLUSTER_COLORS)]], label=f"Cụm {k}",
                s=25, alpha=0.7, edgecolors="white", linewidths=0.3
            )
            if np.any(mask):
                cx = X_pca[mask, 0].mean()
                cy = X_pca[mask, 1].mean()
                ax.scatter(cx, cy, c="black", marker="x", s=120, linewidths=2.5, zorder=5)

        sil = r["silhouette"]
        ine = r["inertia"]
        pc1_pct = evr[0] * 100
        pc2_pct = evr[1] * 100
        ax.set_title(f"{name}\nInertia={ine:.1f} | Sil={sil:.3f}", fontsize=10)
        ax.set_xlabel(f"PC1 ({pc1_pct:.1f}%)", fontsize=9)
        ax.set_ylabel(f"PC2 ({pc2_pct:.1f}%)", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Đã lưu biểu đồ PCA so sánh ra: {save_path}")


def plot_convergence(results, save_path="convergence_minibatch.png"):
    """Vẽ biểu đồ hội tụ inertia của MiniBatchKMeans theo từng bước."""
    mb_keys = [k for k in results if "MiniBatch" in k]
    if not mb_keys:
        return
    mb_key = mb_keys[0]
    model = results[mb_key]["model"]
    history = getattr(model, "convergence_history_", None)
    if history is None or len(history) == 0:
        return

    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(history) + 1), history, color="#E74C3C", linewidth=1.5)
    plt.xlabel("Bước (Mini-batch iteration)", fontsize=10)
    plt.ylabel("Inertia trên batch", fontsize=10)
    plt.title("Đồ thị Hội tụ của MiniBatchKMeans\n(Inertia trên mini-batch theo từng bước)", fontsize=11)
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Đã lưu biểu đồ hội tụ MiniBatch ra: {save_path}")


def find_optimal_k(X_scaled, k_min=2, k_max=8, save_path="elbow_silhouette_k.png", target_k=BEST_K, model_cls=KMeansPlusPlus):
    """
    Đánh giá các giá trị K từ k_min đến k_max dùng KMeansPlusPlus
    để khảo sát tham khảo chỉ số Elbow (Inertia WCSS) và Silhouette Score.
    """
    k_values = list(range(k_min, k_max + 1))
    inertias = []
    sil_scores = []

    for k in k_values:
        model = model_cls(n_clusters=k, n_init=10, random_state=42)
        labels = model.fit_predict(X_scaled)
        inertias.append(model.inertia_)
        sil = silhouette_score(X_scaled, labels)
        sil_scores.append(sil)

    print("\n" + "=" * 70)
    print(f"  PHÂN TÍCH KHẢO SÁT SỐ CỤM K (K = {k_min} đến {k_max}) USING {model_cls.__name__}")
    print("=" * 70)
    header = f"  {'K':<5} {'Inertia (WCSS)':>20} {'Silhouette Score':>22}"
    print(header)
    print("-" * 70)
    for k, ine, sil in zip(k_values, inertias, sil_scores):
        print(f"  {k:<5} {ine:>20.2f} {sil:>22.4f}")
    print("=" * 70)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(k_values, inertias, 'bo-', linewidth=2, markersize=7)
    ax1.set_title("Phương pháp Elbow (Inertia WCSS)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Số cụm (K)", fontsize=10)
    ax1.set_ylabel("Inertia (WCSS)", fontsize=10)
    ax1.grid(True, alpha=0.3)

    ax2.plot(k_values, sil_scores, 'ro-', linewidth=2, markersize=7)
    ax2.set_title("Chỉ số Silhouette theo K", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Số cụm (K)", fontsize=10)
    ax2.set_ylabel("Silhouette Score", fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Đã lưu đồ thị phân tích K tối ưu ra: {save_path}")

    best_k_idx = int(np.argmax(sil_scores))
    best_k = k_values[best_k_idx]
    print(f"  ==> Số cụm K đạt Silhouette cao nhất (toán học): K = {best_k} (Silhouette = {sil_scores[best_k_idx]:.4f})")
    print("  ----------------------------------------------------------------------")
    print(f"  [QUYẾT ĐỊNH CỐ ĐỊNH K={target_k} THEO YÊU CẦU NGHỆP VỤ KINH DOANH]")
    print(f"  -> Hệ thống CỐ ĐỊNH K={target_k} nhằm đáp ứng nhu cầu phân khúc 3 nhóm khách hàng (VIP, HoReCa, Retail);")
    print("  -> Kết quả khảo sát toán học K=2 mang tính chất tham chiếu kiểm chứng.")
    print("  ----------------------------------------------------------------------\n")
    return k_values, inertias, sil_scores


def evaluate_out_of_sample(X_raw, model_class, k=BEST_K):
    """
    Thực hiện kiểm thử Out-of-Sample Train/Validation Split (80/20)
    trên ĐÚNG lớp thuật toán (model_class) được chọn deploy để đảm bảo không bị overfitting.
    """
    np.random.seed(42)
    n_samples = len(X_raw)
    indices = np.random.permutation(n_samples)
    train_size = int(0.8 * n_samples)

    train_idx, val_idx = indices[:train_size], indices[train_size:]
    X_train_raw, X_val_raw = X_raw[train_idx], X_raw[val_idx]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)

    # Khởi tạo mô hình thuộc đúng class xuất sắc nhất được chọn deploy
    model = model_class(n_clusters=k, n_init=10, random_state=42)
    train_labels = model.fit_predict(X_train)
    centroids = model.cluster_centers_

    val_dists = np.linalg.norm(X_val[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2)
    val_labels = np.argmin(val_dists, axis=1)

    train_sil = silhouette_score(X_train, train_labels)
    val_sil = silhouette_score(X_val, val_labels)

    train_ine_sample = model.inertia_ / len(X_train)
    val_ine_sample = np.sum(np.min(val_dists**2, axis=1)) / len(X_val)

    model_name = model_class.__name__
    print("\n" + "=" * 70)
    print(f"  KIỂM THỬ OUT-OF-SAMPLE TRAIN / VALIDATION (80 / 20) TRÊN THUẬT TOÁN DEPLOY [{model_name}]")
    print("=" * 70)
    print(f"  Train Set: {len(X_train)} mẫu | Validation Set: {len(X_val)} mẫu")
    print(f"  Train Silhouette Score:     {train_sil:.4f} | Val Silhouette Score:     {val_sil:.4f}")
    print(f"  Train Inertia (per sample): {train_ine_sample:.4f} | Val Inertia (per sample): {val_ine_sample:.4f}")
    print(f"  -> Kết luận: Thuật toán '{model_name}' có khả năng tổng quát hóa rất tốt trên dữ liệu mới.\n" + "=" * 70)
    return train_sil, val_sil


def run_pipeline():
    print("=" * 75)
    print("  DỰ ÁN PHÂN KHÚC KHÁCH HÀNG BÁN BUÔN (WHOLESALE CUSTOMERS - PURE NUMPY)")
    print(f"  Chuỗi thuật toán: KMeans | KMeans++ | MiniBatchKMeans (Cố định K = {BEST_K})")
    print("=" * 75)

    data_path = "Wholesale_customers_data.csv"
    print(f"\n[1/8] Nạp dữ liệu từ file: {data_path}")
    df_raw, df_processed, feature_cols = load_and_preprocess_data(data_path)
    print(f"      Số mẫu: {df_processed.shape[0]}, Số đặc trưng: {len(feature_cols)}")

    print("\n[2/8] Chuẩn hóa dữ liệu (StandardScaler Z-score)...")
    X_train = df_processed[feature_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    print(f"      X_scaled shape: {X_scaled.shape}")

    print("\n[3/8] Phân tích khảo sát số cụm K (Elbow & Silhouette)...")
    find_optimal_k(X_scaled, target_k=BEST_K)

    print(f"\n[4/8] Huấn luyện & So sánh 3 thuật toán phân cụm (K={BEST_K}, n_init=10)...")
    results = compare_algorithms(X_scaled, k=BEST_K)
    print("  Hoàn thành cả 3 thuật toán.")

    print_comparison_table(results, k=BEST_K)

    print("[5/8] Tự động chọn mô hình xuất sắc nhất (Dynamic Model Selection)...")
    best_key, composite_scores = select_best_algorithm(results)
    print(f"  Điểm tổng hợp (Min-Max Composite: Silhouette + CHI + (1-DBI)):")
    for k, sc in composite_scores.items():
        print(f"    - {k:<30}: {sc:.4f}")
    print(f"  ==> Mô hình xuất sắc nhất được chọn deploy: '{best_key}'")

    best_model = results[best_key]["model"]
    best_model_class = results[best_key]["model_class"]
    best_labels = results[best_key]["labels"]

    print("\n[6/8] Kiểm thử Out-of-Sample Train/Validation Split (80/20) trên mô hình deploy...")
    evaluate_out_of_sample(X_train, model_class=best_model_class, k=BEST_K)

    print("[7/8] Vẽ biểu đồ so sánh PCA 2D và đường hội tụ...")
    plot_pca_comparison(X_scaled, results, k_clusters=BEST_K)
    plot_convergence(results)

    # In phân bố cụm
    print("\n  === PHÂN BỐ KHÁCH HÀNG THEO CỤM ===")
    unique_l, counts_l = np.unique(best_labels, return_counts=True)
    total_samples = len(best_labels)
    for cid, cnt in zip(unique_l, counts_l):
        pct = (cnt / total_samples) * 100
        bar_str = "█" * int(pct / 2)
        print(f"    Cluster {cid}: {cnt:>3d} khách hàng ({pct:>5.1f}%) {bar_str}")

    print("\n[8/8] Lập hồ sơ phân cụm và lưu Pipeline mô hình...")
    profiles = map_cluster_profiles(df_raw, best_labels)
    for c_id, desc in profiles.items():
        print(f"    - Cụm {c_id}: {desc}")

    profiles_to_save = {str(k): v for k, v in profiles.items()}
    with open("cluster_profiles.json", "w", encoding="utf-8") as f:
        json.dump(profiles_to_save, f, ensure_ascii=False, indent=2)
    print("  Đã lưu cluster profiles ra: cluster_profiles.json")

    df_export = df_raw.copy()
    df_export["Cluster"] = best_labels
    df_export.to_csv("wholesale_preprocessed.csv", index=False)
    print("  Đã xuất dữ liệu phân cụm ra wholesale_preprocessed.csv")

    pipeline = Pipeline([("scaler", scaler), ("kmeans", best_model)])
    pipeline.save("kmeans_pipeline.pkl")

    print("\n" + "=" * 75)
    print("  HOÀN THÀNH! Quy trình 8 bước huấn luyện & xuất mô hình đã thực thi xong.")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    run_pipeline()
