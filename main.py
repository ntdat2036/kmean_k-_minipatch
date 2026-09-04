"""
main.py
=======
Script chinh thuc thi quy trinh phan khuc khach hang Wholesale Customers.
TUYET DOI KHONG SU DUNG scikit-learn.
"""

import sys
import os
import time
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
    Pipeline
)

CLUSTER_COLORS = ["#E74C3C", "#2ECC71", "#3498DB"]


def load_and_preprocess_data(data_path):
    """Nap du lieu va tao dac trung (dung log1p cho ca ratio de giam skewness/outliers)."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Khong tim thay file: {data_path}")
    df_raw = pd.read_csv(data_path)
    df = df_raw.copy()
    feature_cols = ["Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicassen"]
    for col in feature_cols:
        df[f"{col}_log"] = np.log1p(df[col])
    df["Total_Spend"] = df[feature_cols].sum(axis=1)

    # Log1p tren cac ty le chi tieu de tranh phinh to Z-score do outliers
    df["Fresh_Ratio_log"]        = np.log1p(df["Fresh"] / (df["Total_Spend"] + 1e-6))
    df["NonEssential_Ratio_log"] = np.log1p((df["Grocery"] + df["Detergents_Paper"]) / (df["Total_Spend"] + 1e-6))
    df["Grocery_Milk_Ratio_log"] = np.log1p(df["Grocery"] / (df["Milk"] + 1e-6))

    training_cols = [f"{col}_log" for col in feature_cols] + [
        "Fresh_Ratio_log", "NonEssential_Ratio_log", "Grocery_Milk_Ratio_log"
    ]
    return df_raw, df, training_cols


def evaluate_clustering(X_scaled, labels):
    """Tinh 3 chi so danh gia: Silhouette, CHI, DBI."""
    return {
        "silhouette": silhouette_score(X_scaled, labels),
        "calinski":   calinski_harabasz_score(X_scaled, labels),
        "davies":     davies_bouldin_score(X_scaled, labels),
    }


def compare_algorithms(X_scaled):
    """
    Huan luyen va so sanh 3 thuat toan phan cum tren cung du lieu (n_init=10 dong nhat).

    Chuoi cai tien:
      KMeans (random) -> KMeansPlusPlus (D^2 init) -> MiniBatchKMeans (online)
    """
    algorithms = {
        "KMeans (Random Init)": KMeans(
            n_clusters=3, n_init=10, random_state=42
        ),
        "KMeans++ (D^2+Oversampling)": KMeansPlusPlus(
            n_clusters=3, n_init=10, random_state=42, oversample_factor=3
        ),
        "MiniBatchKMeans (Online)": MiniBatchKMeans(
            n_clusters=3, n_init=10, random_state=42,
            batch_size=100, max_no_improvement=15
        ),
    }
    results = {}
    for name, model in algorithms.items():
        print(f"    Dang chay: {name} (n_init=10)...")
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
    Tu dong chon thuat toan tot nhat dua tren Composite Metric Score:
    Score = Norm(Silhouette) + Norm(CHI) + Norm(1 - DBI)
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


def print_comparison_table(results):
    """In bang so sanh ket qua 3 thuat toan."""
    sep = "=" * 88
    print("\n" + sep)
    print("  BANG SO SANH 3 THUAT TOAN PHAN CUM (K = 3, n_init = 10)")
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


def plot_pca_comparison(X_scaled, results, save_path="comparison_pca.png"):
    """Ve 3 bieu do PCA 2D so sanh ket qua phan cum."""
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    evr = pca.explained_variance_ratio_

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("So sanh Ket qua Phan cum PCA 2D - 3 Thuat toan",
                 fontsize=14, fontweight="bold")

    for ax, (name, r) in zip(axes, results.items()):
        labels = r["labels"]
        for k in range(3):
            mask = (labels == k)
            ax.scatter(
                X_pca[mask, 0], X_pca[mask, 1],
                c=CLUSTER_COLORS[k], label=f"Cum {k}",
                s=25, alpha=0.7, edgecolors="white", linewidths=0.3
            )
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
    mb_key = [k for k in results if "MiniBatch" in k][0]
    history = results[mb_key]["model"].convergence_history_
    if not history:
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


def run_pipeline():
    print("=" * 70)
    print("  DU AN PHAN KHUC KHACH HANG BAN BUON (PURE NUMPY)")
    print("  So sanh: KMeans | KMeans++ | MiniBatchKMeans")
    print("=" * 70)

    data_path = "Wholesale customers data.csv"
    print(f"\n[1/6] Nap du lieu tu file: {data_path}")
    df_raw, df_processed, feature_cols = load_and_preprocess_data(data_path)
    print(f"      So mau: {df_processed.shape[0]}, So dac trung: {len(feature_cols)}")

    print("\n[2/6] Chuan hoa du lieu (StandardScaler Z-score)...")
    X_train = df_processed[feature_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    print(f"      X_scaled shape: {X_scaled.shape}")

    print("\n[3/6] Huan luyen ca 3 thuat toan phan cum...")
    results = compare_algorithms(X_scaled)
    print("  Hoan thanh ca 3 thuat toan.")

    print_comparison_table(results)

    print("[5/6] Ve bieu do so sanh PCA va hoi tu...")
    plot_pca_comparison(X_scaled, results)
    plot_convergence(results)

    print("\n[6/6] Tu dong chon mo hinh tot nhat (Dynamic Model Selection)...")
    best_key, composite_scores = select_best_algorithm(results)
    print(f"  Diem Composite Metrics (Silhouette + CHI + DBI):")
    for k, sc in composite_scores.items():
        print(f"    - {k:<30}: {sc:.4f}")
    print(f"  ==> Mo hinh duoc chon: '{best_key}'")

    best_labels = results[best_key]["labels"]

    df_export = df_raw.copy()
    df_export["Cluster"] = best_labels
    df_export.to_csv("wholesale_preprocessed.csv", index=False)
    print("  Da xuat du lieu phan cum ra wholesale_preprocessed.csv")

    pipeline = Pipeline([("scaler", scaler), ("kmeans", results[best_key]["model"])])
    pipeline.save("kmeans_pipeline.pkl")

    print("\n" + "=" * 70)
    print("  HOAN THANH! Ket qua da duoc luu thanh cong.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_pipeline()
