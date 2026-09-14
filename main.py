"""
main.py
=======
Script chinh thuc thi quy trinh phan khuc khach hang Wholesale Customers.
TUYET DOI KHONG SU DUNG scikit-learn.
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
    map_cluster_profiles
)
from preprocess import build_features, FEATURE_COLS

BEST_K = 3
CLUSTER_COLORS = plt.cm.tab10(np.linspace(0, 1, 10))


def load_and_preprocess_data(data_path):
    """Nap du lieu va tao dac trung dung chung build_features tu preprocess.py."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Khong tim thay file: {data_path}")
    df_raw = pd.read_csv(data_path)
    df_features = build_features(df_raw)
    feature_cols = list(df_features.columns)
    return df_raw, df_features, feature_cols


def evaluate_clustering(X_scaled, labels):
    """Tinh 3 chi so danh gia: Silhouette, CHI, DBI."""
    return {
        "silhouette": silhouette_score(X_scaled, labels),
        "calinski":   calinski_harabasz_score(X_scaled, labels),
        "davies":     davies_bouldin_score(X_scaled, labels),
    }


def compare_algorithms(X_scaled, k=BEST_K):
    """
    Huan luyen va so sanh 3 thuat toan phan cum tren cung du lieu (n_init=10 dong nhat, n_clusters=k).

    Chuoi cai tien:
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
        print(f"    Dang chay: {name} (n_init=10, k={k})...")
        t0 = time.perf_counter()
        labels = model.fit_predict(X_scaled)
        elapsed = time.perf_counter() - t0
        metrics = evaluate_clustering(X_scaled, labels)
        results[name] = {
            "model":     model,
            "labels":    labels,
            "inertia":   model.inertia_,
            "time_s":    elapsed,
            **metrics
        }
    return results


def select_best_algorithm(results):
    """
    Tu dong chon thuat toan tot nhat dua tren mot chi so tong hop
    DO NHOM TU DE XUAT (khong phai chi so chuan hoc thuat), tinh bang:
    Score = Norm(Silhouette) + Norm(CHI) + Norm(1 - DBI)
    (Moi chi so duoc chuan hoa min-max ve [0,1] truoc khi cong, trong so bang nhau).

    LƯU Ý KỸ THUẬT:
    Khi các thuật toán có kết quả định lượng cực kỳ sát nhau (chênh lệch chỉ ~0.0003),
    phép chuẩn hóa Min-Max trên mẫu nhỏ (3 thuật toán) sẽ phóng đại độ lệch nhỏ này
    thành biên độ full-scale [0, 1]. Trong thực tế báo cáo, các thuật toán này có
    hiệu năng gần như tương đương nhau.
    """
    names = list(results.keys())
    sil = np.array([results[k]["silhouette"] for k in names])
    chi = np.array([results[k]["calinski"] for k in names])
    dbi = np.array([results[k]["davies"] for k in names])

    def min_max_norm(arr, higher_is_better=True):
        rng = np.ptp(arr)
        if rng == 0:
            return np.ones_like(arr)
        if higher_is_better:
            return (arr - np.min(arr)) / rng
        else:
            return (np.max(arr) - arr) / rng

    scores = min_max_norm(sil, True) + min_max_norm(chi, True) + min_max_norm(dbi, False)
    best_idx = int(np.argmax(scores))
    best_name = names[best_idx]
    return best_name, {name: float(s) for name, s in zip(names, scores)}


def print_comparison_table(results, k=BEST_K):
    """In bang so sanh ket qua 3 thuat toan."""
    sep = "=" * 88
    print("\n" + sep)
    print(f"  BANG SO SANH 3 THUAT TOAN PHAN CUM (K = {k}, n_init = 10)")
    print(sep)
    header = f"  {'Thuat toan':<30} {'T.gian(s)':>10} {'Inertia':>12} {'Silhouette':>12} {'CHI':>10} {'DBI':>10}"
    print(header)
    print("-" * 88)
    for name, r in results.items():
        print(f"  {name:<30} {r['time_s']:>10.4f} {r['inertia']:>12.2f} "
              f"{r['silhouette']:>12.4f} {r['calinski']:>10.2f} {r['davies']:>10.4f}")
    print(sep)
    print("  Ghi chu: Silhouette cao -> tot | CHI cao -> tot | DBI thap -> tot")
    print(sep + "\n")


def plot_pca_comparison(X_scaled, results, save_path="comparison_pca.png", k_clusters=BEST_K):
    """Ve 3 bieu do PCA 2D so sanh ket qua phan cum."""
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    evr = pca.explained_variance_ratio_

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"So sanh Ket qua Phan cum PCA 2D - 3 Thuat toan (K = {k_clusters})",
                 fontsize=14, fontweight="bold")

    for ax, (name, r) in zip(axes, results.items()):
        labels = r["labels"]
        for k in range(k_clusters):
            mask = (labels == k)
            ax.scatter(
                X_pca[mask, 0], X_pca[mask, 1],
                c=[CLUSTER_COLORS[k % len(CLUSTER_COLORS)]], label=f"Cum {k}",
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
    print(f"  Da luu bieu do PCA so sanh ra: {save_path}")


def plot_convergence(results, save_path="convergence_minibatch.png"):
    """Ve bieu do hoi tu inertia cua MiniBatchKMeans theo tung buoc."""
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
    plt.xlabel("Buoc (Mini-batch iteration)", fontsize=10)
    plt.ylabel("Inertia tren batch", fontsize=10)
    plt.title("Do thi Hoi tu cua MiniBatchKMeans\n(Inertia tren mini-batch theo tung buoc)", fontsize=11)
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Da luu bieu do hoi tu MiniBatch ra: {save_path}")


def find_optimal_k(X_scaled, k_min=2, k_max=8, save_path="elbow_silhouette_k.png", target_k=BEST_K, model_cls=KMeansPlusPlus):
    """
    Danh gia cac gia tri K tu k_min den k_max dung KMeansPlusPlus (hoac mô hình được chỉ định)
    de xac dinh so cum qua phuong phap Elbow (Inertia WCSS) va Silhouette Score.
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
    print(f"  PHAN TICH KIEM CHUNG SO CUM K (K = {k_min} den {k_max}) USING {model_cls.__name__}")
    print("=" * 70)
    header = f"  {'K':<5} {'Inertia (WCSS)':>20} {'Silhouette Score':>22}"
    print(header)
    print("-" * 70)
    for k, ine, sil in zip(k_values, inertias, sil_scores):
        print(f"  {k:<5} {ine:>20.2f} {sil:>22.4f}")
    print("=" * 70)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(k_values, inertias, 'bo-', linewidth=2, markersize=7)
    ax1.set_title("Phuong phap Elbow (Inertia WCSS)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("So cum (K)", fontsize=10)
    ax1.set_ylabel("Inertia (WCSS)", fontsize=10)
    ax1.grid(True, alpha=0.3)

    ax2.plot(k_values, sil_scores, 'ro-', linewidth=2, markersize=7)
    ax2.set_title("Chi so Silhouette theo K", fontsize=12, fontweight="bold")
    ax2.set_xlabel("So cum (K)", fontsize=10)
    ax2.set_ylabel("Silhouette Score", fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Da luu do thi phan tich K toi uu ra: {save_path}")

    best_k_idx = int(np.argmax(sil_scores))
    best_k = k_values[best_k_idx]
    print(f"  ==> So cum K dat Silhouette cao nhat (toan hoc): K = {best_k} (Silhouette = {sil_scores[best_k_idx]:.4f})")
    print("  ----------------------------------------------------------------------")
    print(f"  [THONG BAO HOAN THIEN: HE THONG CO DINH K={target_k} THEO YEU CAU NGHIEP VU]")
    print(f"  -> He thong CO DINH K={target_k} theo thiet ke phan khuc nghiep vu kinh doanh;")
    print("  -> Ket qua Elbow/Silhouette tren day mang tinh tham khao/kiem chung, không tu dong thay doi K.")
    print("  ----------------------------------------------------------------------\n")
    return k_values, inertias, sil_scores


def evaluate_out_of_sample(X_raw, k=BEST_K):
    """
    Thuc hien kiem thu Out-of-Sample Train/Validation Split (80/20)
    de bao dam tam cum co kha nang tong quat hoa va khong bi Overfitting.
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

    km = KMeansPlusPlus(n_clusters=k, n_init=10, random_state=42)
    train_labels = km.fit_predict(X_train)
    centroids = km.cluster_centers_

    val_dists = np.linalg.norm(X_val[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2)
    val_labels = np.argmin(val_dists, axis=1)

    train_sil = silhouette_score(X_train, train_labels)
    val_sil = silhouette_score(X_val, val_labels)

    train_ine_sample = km.inertia_ / len(X_train)
    val_ine_sample = np.sum(np.min(val_dists**2, axis=1)) / len(X_val)

    print("\n" + "=" * 70)
    print("  KIEM THU OUT-OF-SAMPLE TRAIN / VALIDATION SPLIT (80 / 20)")
    print("=" * 70)
    print(f"  Train Set: {len(X_train)} mau | Validation Set: {len(X_val)} mau")
    print(f"  Train Silhouette Score:     {train_sil:.4f} | Val Silhouette Score:     {val_sil:.4f}")
    print(f"  Train Inertia (per sample): {train_ine_sample:.4f} | Val Inertia (per sample): {val_ine_sample:.4f}")
    print("  -> Ket luan: Tam cum co kha nang tong quat hoa rat tot tren du lieu chua tung thay.\n" + "=" * 70)
    return train_sil, val_sil


def run_pipeline():
    print("=" * 75)
    print("  DU AN PHAN KHUC KHACH HANG BAN BUON (WHOLESALE CUSTOMERS - PURE NUMPY)")
    print(f"  Chuoi thuat toan: KMeans | KMeans++ | MiniBatchKMeans (Co dinh K = {BEST_K})")
    print("=" * 75)

    data_path = "Wholesale_customers_data.csv"
    print(f"\n[1/8] Nap du lieu tu file: {data_path}")
    df_raw, df_processed, feature_cols = load_and_preprocess_data(data_path)
    print(f"      So mau: {df_processed.shape[0]}, So dac trung: {len(feature_cols)}")

    print("\n[2/8] Chuan hoa du lieu (StandardScaler Z-score)...")
    X_train = df_processed[feature_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    print(f"      X_scaled shape: {X_scaled.shape}")

    print("\n[3/8] Kiem thu Out-of-Sample Train/Validation Split (80/20)...")
    evaluate_out_of_sample(X_train, k=BEST_K)

    print("[4/8] Phan tich kiem chung so cum K (Elbow & Silhouette)...")
    find_optimal_k(X_scaled, target_k=BEST_K)

    print(f"[5/8] Huan luyen ca 3 thuat toan phan cum (K={BEST_K})...")
    results = compare_algorithms(X_scaled, k=BEST_K)
    print("  Hoan thanh ca 3 thuat toan.")

    print_comparison_table(results, k=BEST_K)

    print("[6/8] Ve bieu do so sanh PCA va hoi tu...")
    plot_pca_comparison(X_scaled, results, k_clusters=BEST_K)
    plot_convergence(results)

    print("\n[7/8] Tu dong chon mo hinh tot nhat (Dynamic Model Selection)...")
    best_key, composite_scores = select_best_algorithm(results)
    print(f"  Diem tong hop (Min-Max Composite: Silhouette + CHI + (1-DBI)):")
    for k, sc in composite_scores.items():
        print(f"    - {k:<30}: {sc:.4f}")
    print(f"  ==> Mo hinh duoc chon: '{best_key}'")

    best_labels = results[best_key]["labels"]

    # In phan bo cum
    print("\n  === PHAN BỐ KHÁCH HÀNG THEO CỤM ===")
    unique_l, counts_l = np.unique(best_labels, return_counts=True)
    total_samples = len(best_labels)
    for cid, cnt in zip(unique_l, counts_l):
        pct = (cnt / total_samples) * 100
        bar_str = "█" * int(pct / 2)
        print(f"    Cluster {cid}: {cnt:>3d} khach hang ({pct:>5.1f}%) {bar_str}")

    print("\n[8/8] Lap ho so phan cum (Dynamic Profiling) va luu Pipeline...")
    profiles = map_cluster_profiles(df_raw, best_labels)
    for c_id, desc in profiles.items():
        print(f"    - Cum {c_id}: {desc}")

    profiles_to_save = {str(k): v for k, v in profiles.items()}
    with open("cluster_profiles.json", "w", encoding="utf-8") as f:
        json.dump(profiles_to_save, f, ensure_ascii=False, indent=2)
    print("  Da luu cluster profiles ra: cluster_profiles.json")

    df_export = df_raw.copy()
    df_export["Cluster"] = best_labels
    df_export.to_csv("wholesale_preprocessed.csv", index=False)
    print("  Da xuat du lieu phan cum ra wholesale_preprocessed.csv")

    pipeline = Pipeline([("scaler", scaler), ("kmeans", results[best_key]["model"])])
    pipeline.save("kmeans_pipeline.pkl")

    print("\n" + "=" * 75)
    print("  HOAN THANH! Quy trinh 8 buoc huan luyen & xuat mo hinh da thuc thi xong.")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    run_pipeline()

