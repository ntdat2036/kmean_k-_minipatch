"""
model.py
========
Thư viện thuật toán Machine Learning tự viết hoàn toàn bang NumPy.
TUYET DOI KHONG SU DUNG scikit-learn (sklearn).

Cac thanh phan:
  - StandardScaler          : Chuan hoa Z-score
  - KMeans                  : Phan cum K-Means Lloyd (khoi tao ngau nhien - RANDOM INIT)
  - KMeansPlusPlus          : Phan cum K-Means++ (khoi tao D^2-weighted + oversampling)
  - MiniBatchKMeans         : Phan cum Mini-Batch K-Means (cap nhat online theo batch)
  - PCA                     : Giam chieu (SVD)
  - silhouette_score        : Chi so Silhouette
  - calinski_harabasz_score : Chi so Calinski-Harabasz
  - davies_bouldin_score    : Chi so Davies-Bouldin
  - Pipeline                : Dong goi quy trinh xu ly + phan cum

Chuoi cai tien thuat toan:
  KMeans (ngau nhien) -> KMeansPlusPlus (init D^2) -> MiniBatchKMeans (online mini-batch)
"""

import numpy as np
import pandas as pd
import pickle


# ===========================================================================
# CLASS: StandardScaler
# ===========================================================================

class StandardScaler:
    """
    Chuẩn hóa đặc trưng theo Z-score (mean = 0, std = 1).

    Công thức toán học:
    --------------------
        Bước FIT:
            mu_j = (1/n) * sum(x_ij)          (trung bình cột j)
            sigma_j = sqrt((1/n) * sum((x_ij - mu_j)^2))   (độ lệch chuẩn cột j, ddof=0)

        Bước TRANSFORM:
            z_ij = (x_ij - mu_j) / sigma_j

        Bước INVERSE TRANSFORM:
            x_ij = z_ij * sigma_j + mu_j

    Thuộc tính sau fit:
    -------------------
        mean_  : ndarray, shape (n_features,)  - vector trung bình từng cột.
        scale_ : ndarray, shape (n_features,)  - vector độ lệch chuẩn từng cột.
    """

    def __init__(self):
        self.mean_ = None    # mu - vector trung bình
        self.scale_ = None   # sigma - vector độ lệch chuẩn

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0, ddof=0)
        self.scale_[self.scale_ == 0.0] = 1.0
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, Z):
        Z = np.asarray(Z, dtype=np.float64)
        return Z * self.scale_ + self.mean_


# ===========================================================================
# CLASS: KMeans
# ===========================================================================

class KMeans:
    """
    Thuật toán phân cụm K-Means Lloyd với khởi tạo NGAU NHIEN (Random Init).

    THIET SOT cua KMeans ngau nhien:
    ---------------------------------
    - Chon tam dau tien hoan toan ngau nhien -> de roi vao cuc tri cuc bo xau.
    - Cac tam co the khoi tao gan nhau -> mat nhieu vong lap Lloyd.
    - Khong co bao dam ly thuyet ve chat luong nghiem toi uu.
    => Giai phap: KMeansPlusPlus (khoi tao D^2-weighted).

    Công thức toán học:
    --------------------
    Hàm mục tiêu — tối thiểu hóa WCSS (Within-Cluster Sum of Squares):
        J = sum_{k=1}^{K} sum_{i in C_k} || x_i - mu_k ||^2

    Khởi tạo ngẫu nhiên:
        idx_k ~ Uniform({1,...,n}), chon K chi so khong trung lap.

    Thuật toán Lloyd (lặp đến hội tụ):
      E-step (gán nhãn):
            C_k = { x_i : k = argmin_j || x_i - mu_j ||^2 }
      M-step (cập nhật tâm):
            mu_k = (1 / |C_k|) * sum_{i in C_k} x_i

    Điều kiện hội tụ:
        Delta = sum_{k=1}^{K} || mu_k^new - mu_k^old ||^2 <= tol

    Tham số (Hyperparameters):
    --------------------------
        n_clusters   : int   — so cum K. Mac dinh = 3.
        max_iter     : int   — so vong lap Lloyd toi da. Mac dinh = 300.
        n_init       : int   — so lan khoi tao lai. Mac dinh = 10.
        random_state : int   — seed ngau nhien. Mac dinh = None.
        tol          : float — nguong hoi tu. Mac dinh = 1e-4.

    Thuộc tính sau fit:
    -------------------
        cluster_centers_ : ndarray (K, n_features) — toa do tam cum.
        labels_          : ndarray (n_samples,)     — nhan cum tung diem.
        inertia_         : float                    — WCSS cuoi cung.
    """

    def __init__(self, n_clusters=3, max_iter=300, n_init=10,
                 random_state=None, tol=1e-4, **kwargs):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.n_init = n_init
        self.random_state = random_state
        self.tol = tol

        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None

    def _random_init(self, X, rng):
        """
        Khởi tạo ngẫu nhiên đều: chọn K điểm từ X không lặp.
            idx ~ Uniform({0,...,n-1}), size=K, replace=False
        """
        n_samples = X.shape[0]
        indices = rng.choice(n_samples, size=self.n_clusters, replace=False)
        return X[indices].copy()

    def _compute_distances(self, X, centers):
        """
        Khoảng cách Euclidean n điểm đến K tâm:
            dist[i, k] = || x_i - mu_k ||_2
        """
        diff = X[:, np.newaxis, :] - centers[np.newaxis, :, :]
        return np.sqrt(np.sum(diff ** 2, axis=2))

    def _compute_inertia(self, X, centers, labels):
        """
        WCSS: J = sum_{i=1}^{n} || x_i - mu_{labels[i]} ||^2
        """
        inertia = 0.0
        for k in range(self.n_clusters):
            mask = (labels == k)
            if np.any(mask):
                inertia += np.sum((X[mask] - centers[k]) ** 2)
        return inertia

    def _fit_single(self, X, rng):
        n_samples = X.shape[0]
        centers = self._random_init(X, rng)
        labels = np.full(n_samples, -1, dtype=int)

        for _ in range(self.max_iter):
            # E-step: gan nhan theo tam gan nhat
            distances = self._compute_distances(X, centers)
            new_labels = np.argmin(distances, axis=1)

            # M-step: cap nhat tam
            new_centers = np.zeros_like(centers)
            for k in range(self.n_clusters):
                pts = X[new_labels == k]
                new_centers[k] = np.mean(pts, axis=0) if len(pts) > 0 else centers[k]

            shift = np.sum((new_centers - centers) ** 2)
            centers = new_centers
            labels = new_labels
            if shift <= self.tol:
                break

        return centers, labels, self._compute_inertia(X, centers, labels)

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        base_seed = self.random_state if self.random_state is not None else np.random.randint(0, 100_000)

        best_inertia = np.inf
        best_centers = best_labels = None

        for run in range(self.n_init):
            rng = np.random.RandomState(base_seed + run)
            centers, labels, inertia = self._fit_single(X, rng)
            if inertia < best_inertia:
                best_inertia = inertia
                best_centers = centers.copy()
                best_labels = labels.copy()

        self.cluster_centers_ = best_centers
        self.labels_ = best_labels
        self.inertia_ = best_inertia
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        return np.argmin(self._compute_distances(X, self.cluster_centers_), axis=1)

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_


# ===========================================================================
# CLASS: KMeansPlusPlus  (Khởi tạo D^2-weighted + Oversampling)
# ===========================================================================

class KMeansPlusPlus:
    """
    Thuật toán K-Means++ với khởi tạo D^2-weighted và oversampling.

    Cải tiến so với KMeans (ngẫu nhiên):
    ---------------------------------------
    - Khởi tạo tâm phân bổ đều trong không gian đặc trưng theo D^2-weighting.
    - Bảo đảm lý thuyết (Arthur & Vassilvitskii, 2007):
          E[WCSS] <= O(log K) * WCSS_optimal
    - Oversampling: mỗi bước thử l = oversample_factor ứng viên, chọn ứng viên
      giảm potential nhiều nhất (tốt hơn KMeans++ gốc chọn 1 ứng viên ngẫu nhiên).

    Thiếu sót còn lại:
    -------------------
    - Mỗi vòng lặp Lloyd vẫn duyệt TOAN BO n mẫu -> O(n * K * d * T).
    - Không scalable với dữ liệu rất lớn.
    => Giải pháp: MiniBatchKMeans.

    Công thức toán học — Khởi tạo D^2-weighted:
    ---------------------------------------------
    Bước 1: Chọn tâm mu_1 ngẫu nhiên đều.

    Bước k (2 -> K):
        D(x_i)  = min_{j<k} || x_i - mu_j ||_2
        P(x_i)  = D(x_i)^2 / sum_{j=1}^{n} D(x_j)^2

    Oversampling (l = oversample_factor ứng viên mỗi bước):
        {c_1,...,c_l} ~ P(x_i)  (sampling l ung vien doc lap)
        phi(c) = sum_{i=1}^{n} min(D(x_i)^2, ||x_i - c||^2)  (potential neu chon c)
        mu_k   = argmin_{c in {c_1,...,c_l}} phi(c)

    Lloyd (giống KMeans):
        E-step: label_i = argmin_k || x_i - mu_k ||^2
        M-step: mu_k = mean(x_i : label_i = k)

    Tham số (Hyperparameters):
    --------------------------
        n_clusters       : int   — so cum K. Mac dinh = 3.
        max_iter         : int   — so vong lap Lloyd toi da. Mac dinh = 300.
        n_init           : int   — so lan khoi tao lai. Mac dinh = 5.
        random_state     : int   — seed ngau nhien. Mac dinh = None.
        tol              : float — nguong hoi tu. Mac dinh = 1e-4.
        oversample_factor: int   — so ung vien moi buoc init. Mac dinh = 3.

    Thuộc tính sau fit:
    -------------------
        cluster_centers_ : ndarray (K, n_features).
        labels_          : ndarray (n_samples,).
        inertia_         : float — WCSS tot nhat trong n_init lan.
        inertia_history_ : list  — WCSS cua tung lan chay n_init.
    """

    def __init__(self, n_clusters=3, max_iter=300, n_init=5,
                 random_state=None, tol=1e-4, oversample_factor=3, **kwargs):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.n_init = n_init
        self.random_state = random_state
        self.tol = tol
        self.oversample_factor = oversample_factor

        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None
        self.inertia_history_ = []

    def _d2_init_with_oversampling(self, X, rng):
        """
        Khởi tạo K-Means++ với oversampling:
            Moi buoc k: chon l ung vien theo P(x_i)=D^2(x_i)/sumD^2,
            giu ung vien co phi(c) = sum_i min(D^2(x_i), ||x_i-c||^2) nho nhat.
        """
        n_samples, n_features = X.shape
        centers = np.empty((self.n_clusters, n_features), dtype=np.float64)

        selected_indices = set()
        first_idx = rng.randint(0, n_samples)
        selected_indices.add(first_idx)
        centers[0] = X[first_idx]
        dist_sq = np.sum((X - centers[0]) ** 2, axis=1)

        for k in range(1, self.n_clusters):
            probs = dist_sq.copy()
            for s_idx in selected_indices:
                probs[s_idx] = 0.0
            sum_dist = np.sum(probs)
            if sum_dist > 0.0:
                probs /= sum_dist
            else:
                probs = np.ones(n_samples, dtype=np.float64)
                for s_idx in selected_indices:
                    probs[s_idx] = 0.0
                probs /= (np.sum(probs) + 1e-12)

            cumprobs = np.cumsum(probs)

            # Oversampling: chon l ung vien khong trung voi tam da chon
            l = min(n_samples - len(selected_indices), max(1, self.oversample_factor))
            candidate_idxs = []
            for _ in range(l):
                r = rng.rand()
                idx = min(np.searchsorted(cumprobs, r), n_samples - 1)
                candidate_idxs.append(idx)

            # Chon ung vien co phi(c) nho nhat
            best_idx = candidate_idxs[0]
            best_phi = np.inf
            for cidx in candidate_idxs:
                c = X[cidx]
                new_d = np.sum((X - c) ** 2, axis=1)
                phi_c = float(np.sum(np.minimum(dist_sq, new_d)))
                if phi_c < best_phi:
                    best_phi = phi_c
                    best_idx = cidx

            selected_indices.add(best_idx)
            centers[k] = X[best_idx]
            if k < self.n_clusters - 1:
                new_d = np.sum((X - centers[k]) ** 2, axis=1)
                dist_sq = np.minimum(dist_sq, new_d)

        return centers

    def _compute_distances(self, X, centers):
        diff = X[:, np.newaxis, :] - centers[np.newaxis, :, :]
        return np.sqrt(np.sum(diff ** 2, axis=2))

    def _compute_inertia(self, X, centers, labels):
        inertia = 0.0
        for k in range(self.n_clusters):
            mask = (labels == k)
            if np.any(mask):
                inertia += np.sum((X[mask] - centers[k]) ** 2)
        return inertia

    def _fit_single(self, X, rng):
        n_samples = X.shape[0]
        centers = self._d2_init_with_oversampling(X, rng)
        labels = np.full(n_samples, -1, dtype=int)

        for _ in range(self.max_iter):
            distances = self._compute_distances(X, centers)
            new_labels = np.argmin(distances, axis=1)

            new_centers = np.zeros_like(centers)
            for k in range(self.n_clusters):
                pts = X[new_labels == k]
                new_centers[k] = np.mean(pts, axis=0) if len(pts) > 0 else centers[k]

            shift = np.sum((new_centers - centers) ** 2)
            centers = new_centers
            labels = new_labels
            if shift <= self.tol:
                break

        return centers, labels, self._compute_inertia(X, centers, labels)

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        base_seed = self.random_state if self.random_state is not None else np.random.randint(0, 100_000)

        best_inertia = np.inf
        best_centers = best_labels = None
        self.inertia_history_ = []

        for run in range(self.n_init):
            rng = np.random.RandomState(base_seed + run)
            centers, labels, inertia = self._fit_single(X, rng)
            self.inertia_history_.append(inertia)
            if inertia < best_inertia:
                best_inertia = inertia
                best_centers = centers.copy()
                best_labels = labels.copy()

        self.cluster_centers_ = best_centers
        self.labels_ = best_labels
        self.inertia_ = best_inertia
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        return np.argmin(self._compute_distances(X, self.cluster_centers_), axis=1)

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_


# ===========================================================================
# CLASS: MiniBatchKMeans  (Cập nhật trực tuyến theo mini-batch)
# ===========================================================================

class MiniBatchKMeans:
    """
    Thuật toán Mini-Batch K-Means với cập nhật online theo mini-batch.

    Cải tiến so với KMeansPlusPlus:
    ---------------------------------
    - Khac phuc: Moi vong lap KMeans++ duyet toan bo n mau -> cham voi data lon.
    - Moi iteration chi xu ly batch_size mau ngau nhien -> O(b*K*d), b << n.
    - Cap nhat tam theo online learning rate (giam dan), khong can luu toan bo data.
    - Ho tro dung som (early stopping) khi khong con cai thien.

    Công thức toán học:
    --------------------
    Khởi tạo: K-Means++ D^2-weighted (giống KMeansPlusPlus).

    Mỗi iteration t:
    1. Lay ngau nhien mini-batch:
            B_t ~ Uniform(X, batch_size)   (khong lap)

    2. Gan nhan mini-batch:
            c(x_i) = argmin_k || x_i - mu_k ||^2    voi x_i in B_t

    3. Cap nhat tam online (per sample trong batch):
            n_k    <- n_k + 1              (dem tich luy)
            eta_k   = 1 / n_k             (learning rate giam dan)
            mu_k   <- (1 - eta_k)*mu_k + eta_k*x_i

    4. Inertia tren batch:
            J_batch = sum_{x_i in B_t} min_k || x_i - mu_k ||^2

    Dung som & Tracking best_centers: Dùng full_inertia trên toàn bộ dữ liệu X
    tránh nhiễu do mini-batch ngẫu nhiên.
    Nhan cuoi: gan toan bo X vao tam gan nhat sau toi uu.

    Tham số (Hyperparameters):
    --------------------------
        n_clusters         : int   — so cum K. Mac dinh = 3.
        max_iter           : int   — so buoc mini-batch toi da. Mac dinh = 200.
        batch_size         : int   — kich thuoc moi mini-batch b. Mac dinh = 100.
        random_state       : int   — seed ngau nhien. Mac dinh = None.
        tol                : float — nguong cai thien inertia. Mac dinh = 1e-4.
        max_no_improvement : int   — so buoc khong cai thien truoc khi stop. Mac dinh = 10.
        n_init             : int   — so lan khoi tao lai. Mac dinh = 3.

    Thuộc tính sau fit:
    -------------------
        cluster_centers_    : ndarray (K, n_features).
        labels_             : ndarray (n_samples,).
        inertia_            : float — WCSS cuoi cung (full dataset).
        convergence_history_: list  — inertia batch tung buoc (lan tot nhat).
        n_iter_             : int   — so buoc thuc su thuc hien.
    """

    def __init__(self, n_clusters=3, max_iter=200, batch_size=100,
                 random_state=None, tol=1e-4, max_no_improvement=10, n_init=3, **kwargs):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.random_state = random_state
        self.tol = tol
        self.max_no_improvement = max_no_improvement
        self.n_init = n_init

        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None
        self.convergence_history_ = []
        self.n_iter_ = 0

    def _kmeans_pp_init(self, X, rng):
        """
        Khởi tạo K-Means++ D^2-weighted (loại trừ trùng lặp ứng viên):
            P(x_i) = D(x_i)^2 / sum_j D(x_j)^2
        """
        n_samples, n_features = X.shape
        centers = np.empty((self.n_clusters, n_features), dtype=np.float64)

        selected_indices = set()
        first_idx = rng.randint(0, n_samples)
        selected_indices.add(first_idx)
        centers[0] = X[first_idx]
        dist_sq = np.sum((X - centers[0]) ** 2, axis=1)

        for k in range(1, self.n_clusters):
            probs = dist_sq.copy()
            for s_idx in selected_indices:
                probs[s_idx] = 0.0
            sum_dist = np.sum(probs)
            if sum_dist > 0.0:
                probs /= sum_dist
            else:
                probs = np.ones(n_samples, dtype=np.float64)
                for s_idx in selected_indices:
                    probs[s_idx] = 0.0
                probs /= (np.sum(probs) + 1e-12)

            cumprobs = np.cumsum(probs)
            r = rng.rand()
            next_idx = min(np.searchsorted(cumprobs, r), n_samples - 1)
            selected_indices.add(next_idx)
            centers[k] = X[next_idx]
            if k < self.n_clusters - 1:
                new_d = np.sum((X - centers[k]) ** 2, axis=1)
                dist_sq = np.minimum(dist_sq, new_d)

        return centers

    def _assign(self, X, centers):
        """
        Gan nhan: c(x_i) = argmin_k || x_i - mu_k ||^2
        """
        diff = X[:, np.newaxis, :] - centers[np.newaxis, :, :]
        return np.argmin(np.sum(diff ** 2, axis=2), axis=1)

    def _compute_inertia(self, X, centers, labels):
        inertia = 0.0
        for k in range(self.n_clusters):
            mask = (labels == k)
            if np.any(mask):
                inertia += np.sum((X[mask] - centers[k]) ** 2)
        return inertia

    def _fit_single(self, X, rng):
        n_samples, n_features = X.shape
        centers = self._kmeans_pp_init(X, rng)

        # n_k[k]: so lan diem nao do duoc gan vao cum k (tich luy)
        n_k = np.zeros(self.n_clusters, dtype=np.float64)

        convergence_hist = []
        best_inertia = np.inf
        best_centers = centers.copy()
        no_improve = 0
        actual_iter = 0

        for _ in range(self.max_iter):
            actual_iter += 1

            # Buoc 1: Lay ngau nhien mini-batch B_t
            b = min(self.batch_size, n_samples)
            batch_idx = rng.choice(n_samples, size=b, replace=False)
            X_batch = X[batch_idx]

            # Buoc 2: Gan nhan cho tung x_i trong B_t
            labels_batch = self._assign(X_batch, centers)

            # Buoc 3: Cap nhat online theo tung mau
            for i in range(b):
                k = labels_batch[i]
                n_k[k] += 1.0
                eta_k = 1.0 / n_k[k]   # learning rate: eta_k = 1/n_k
                centers[k] = (1.0 - eta_k) * centers[k] + eta_k * X_batch[i]

            # Buoc 4: Tinh inertia tren mini-batch de ve do thi
            diff_b = X_batch[:, np.newaxis, :] - centers[np.newaxis, :, :]
            dist_sq_b = np.sum(diff_b ** 2, axis=2)
            batch_inertia = float(np.sum(np.min(dist_sq_b, axis=1)))
            convergence_hist.append(batch_inertia)

            # Đánh giá full_inertia mỗi 5 bước hoặc ở bước cuối cùng để tối ưu tốc độ và lọc nhiễu batch
            if (actual_iter % 5 == 0) or (_ == self.max_iter - 1):
                full_labels = self._assign(X, centers)
                full_inertia = self._compute_inertia(X, centers, full_labels)

                if full_inertia < best_inertia - self.tol:
                    best_inertia = full_inertia
                    best_centers = centers.copy()
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= self.max_no_improvement:
                        break

        # Nhan cuoi: gan toan bo X vao tam tot nhat
        final_labels = self._assign(X, best_centers)
        final_inertia = self._compute_inertia(X, best_centers, final_labels)

        return best_centers, final_labels, final_inertia, convergence_hist, actual_iter

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        base_seed = self.random_state if self.random_state is not None else np.random.randint(0, 100_000)

        best_inertia = np.inf
        best_centers = best_labels = None
        best_history = []
        best_n_iter = 0

        for run in range(self.n_init):
            rng = np.random.RandomState(base_seed + run)
            centers, labels, inertia, history, n_iter = self._fit_single(X, rng)
            if inertia < best_inertia:
                best_inertia = inertia
                best_centers = centers.copy()
                best_labels = labels.copy()
                best_history = history
                best_n_iter = n_iter

        self.cluster_centers_ = best_centers
        self.labels_ = best_labels
        self.inertia_ = best_inertia
        self.convergence_history_ = best_history
        self.n_iter_ = best_n_iter
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        return self._assign(X, self.cluster_centers_)

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_


# ===========================================================================
# CLASS: PCA
# ===========================================================================

class PCA:
    """
    Phân tích thành phần chính (Principal Component Analysis) dùng SVD.

    Công thức toán học:
    --------------------
    1. Trừ trung bình:
            X_centered = X - mu
    2. SVD:
            X_centered = U * Sigma * V^T
    3. Thành phần chính (Sign Determinism adjustment):
            W = V^T[:n_components] * sign(V^T)
    4. Chiếu dữ liệu:
            Z = X_centered * W^T
    """

    def __init__(self, n_components=2, **kwargs):
        self.n_components = n_components
        self.components_ = None
        self.explained_variance_ratio_ = None
        self.mean_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        n, d = X.shape

        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_

        U, sigma, Vt = np.linalg.svd(X_centered, full_matrices=False)

        # Quyen tac Sign Determinism cho SVD (tra phai ky tu lon nhat co dau duong)
        components = Vt[:self.n_components]
        max_abs_cols = np.argmax(np.abs(components), axis=1)
        signs = np.sign(components[np.arange(self.n_components), max_abs_cols])
        signs[signs == 0] = 1.0
        self.components_ = components * signs[:, np.newaxis]

        explained_variance = (sigma ** 2) / (n - 1)
        total_variance = np.sum(explained_variance)

        if total_variance > 0:
            self.explained_variance_ratio_ = explained_variance[:self.n_components] / total_variance
        else:
            self.explained_variance_ratio_ = np.zeros(self.n_components)

        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        X_centered = X - self.mean_
        return np.dot(X_centered, self.components_.T)

    def fit_transform(self, X):
        return self.fit(X).transform(X)


# ===========================================================================
# METRICS FUNCTIONS
# ===========================================================================

def silhouette_score(X, labels):
    """
    Tính hệ số Silhouette trung bình.
    s(i) = (b(i) - a(i)) / max(a(i), b(i))
    """
    X = np.asarray(X, dtype=np.float64)
    labels = np.asarray(labels)
    unique_labels = np.unique(labels)
    n_samples = X.shape[0]

    if len(unique_labels) < 2 or len(unique_labels) >= n_samples:
        return np.nan

    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(diff ** 2, axis=2))

    s_scores = np.zeros(n_samples, dtype=np.float64)

    for i in range(n_samples):
        own_cluster = labels[i]
        own_mask = (labels == own_cluster)

        n_own = np.sum(own_mask)
        if n_own > 1:
            a_i = (np.sum(dist_matrix[i, own_mask]) - dist_matrix[i, i]) / (n_own - 1)
        else:
            a_i = 0.0

        b_i = np.inf
        for other_cluster in unique_labels:
            if other_cluster == own_cluster:
                continue
            other_mask = (labels == other_cluster)
            n_other = np.sum(other_mask)
            if n_other > 0:
                avg_dist = np.sum(dist_matrix[i, other_mask]) / n_other
                if avg_dist < b_i:
                    b_i = avg_dist

        if b_i == np.inf:
            b_i = 0.0

        denom = max(a_i, b_i)
        s_scores[i] = (b_i - a_i) / denom if denom > 0 else 0.0

    return float(np.mean(s_scores))


def calinski_harabasz_score(X, labels):
    """
    Tính chỉ số Calinski-Harabasz (Variance Ratio Criterion).
    CH = (B / (K - 1)) / (W / (n - K))
    """
    X = np.asarray(X, dtype=np.float64)
    labels = np.asarray(labels)
    unique_labels = np.unique(labels)
    n, d = X.shape
    K = len(unique_labels)

    if K < 2 or K >= n:
        return np.nan

    grand_mean = np.mean(X, axis=0)

    B = 0.0
    W = 0.0

    for label in unique_labels:
        C_k = X[labels == label]
        n_k = C_k.shape[0]
        if n_k == 0:
            continue

        mu_k = np.mean(C_k, axis=0)
        B += n_k * np.sum((mu_k - grand_mean) ** 2)
        W += np.sum((C_k - mu_k) ** 2)

    if W == 0.0:
        # Avoid division by zero when points in clusters are identical to centroids
        return 1.0e12

    ch_score = (B / (K - 1)) / (W / (n - K))
    return float(ch_score)


def davies_bouldin_score(X, labels):
    """
    Tính chỉ số Davies-Bouldin (DB Index).
    DB = (1/K) * sum_{i=1}^{K} max_{j != i} ((s_i + s_j) / d(mu_i, mu_j))
    """
    X = np.asarray(X, dtype=np.float64)
    labels = np.asarray(labels)
    unique_labels = np.unique(labels)
    K = len(unique_labels)
    n = X.shape[0]

    if K < 2 or K >= n:
        return np.nan

    centroids = []
    dispersions = []

    for label in unique_labels:
        C_k = X[labels == label]
        mu_k = np.mean(C_k, axis=0)
        s_k = np.mean(np.linalg.norm(C_k - mu_k, axis=1))
        centroids.append(mu_k)
        dispersions.append(s_k)

    centroids = np.array(centroids)
    dispersions = np.array(dispersions)

    R = np.zeros((K, K), dtype=np.float64)
    for i in range(K):
        for j in range(K):
            if i == j:
                continue
            d_ij = np.linalg.norm(centroids[i] - centroids[j])
            if d_ij > 0:
                R[i, j] = (dispersions[i] + dispersions[j]) / d_ij
            else:
                R[i, j] = 0.0

    np.fill_diagonal(R, -np.inf)
    D = np.max(R, axis=1)
    D[D == -np.inf] = 0.0

    return float(np.mean(D))

    centroids = []
    dispersions = []

    for label in unique_labels:
        C_k = X[labels == label]
        mu_k = np.mean(C_k, axis=0)
        s_k = np.mean(np.linalg.norm(C_k - mu_k, axis=1))
        centroids.append(mu_k)
        dispersions.append(s_k)

    centroids = np.array(centroids)
    dispersions = np.array(dispersions)

    R = np.zeros((K, K), dtype=np.float64)
    for i in range(K):
        for j in range(K):
            if i == j:
                continue
            d_ij = np.linalg.norm(centroids[i] - centroids[j])
            if d_ij > 0:
                R[i, j] = (dispersions[i] + dispersions[j]) / d_ij
            else:
                R[i, j] = 0.0

    np.fill_diagonal(R, -np.inf)
    D = np.max(R, axis=1)
    D[D == -np.inf] = 0.0

    return float(np.mean(D))


# ===========================================================================
# CLASS: Pipeline
# ===========================================================================

class Pipeline:
    """
    Chuỗi xử lý (Pipeline) tự cài đặt: nối tuần tự các bước tiền xử lý + mô hình.
    """

    def __init__(self, steps):
        self.steps = steps
        self.named_steps = {name: step for name, step in steps}

    def fit(self, X):
        data = np.asarray(X, dtype=np.float64)
        for name, step in self.steps[:-1]:
            data = step.fit_transform(data)
        self.steps[-1][1].fit(data)
        return self

    def predict(self, X):
        data = np.asarray(X, dtype=np.float64)
        for name, step in self.steps[:-1]:
            data = step.transform(data)
        return self.steps[-1][1].predict(data)

    def fit_predict(self, X):
        self.fit(X)
        return self.predict(X)

    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"[Pipeline] Đã lưu pipeline vào: {filepath}")

    @staticmethod
    def load(filepath):
        with open(filepath, 'rb') as f:
            pipeline = pickle.load(f)
        print(f"[Pipeline] Đã nạp pipeline từ: {filepath}")
        return pipeline


# ===========================================================================
# SHARED PREPROCESSING & DYNAMIC CLUSTER PROFILING
# ===========================================================================

from preprocess import validate_input_data, build_features

preprocess_features = build_features


def map_cluster_profiles(df_raw, labels):
    """
    Ánh xạ động nhãn cụm (0, 1, 2) dựa trên đặc trưng chi tiêu thực tế thay vì hard-code index:
    - Cụm có Total Spend trung bình cao nhất -> "VIP / Cao cấp"
    - Trong các cụm còn lại, cụm có tỷ lệ Fresh + Frozen cao nhất -> "HoReCa (Nhà hàng / Khách sạn)"
    - Cụm còn lại -> "Retail (Bán lẻ phổ thông)"
    """
    df_temp = df_raw.copy()
    df_temp['Cluster'] = labels
    feature_cols = ['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen']
    df_temp['Total_Spend'] = df_temp[feature_cols].sum(axis=1)

    cluster_ids = np.unique(labels)
    profiles = {}

    # 1. Tìm cụm VIP: Total Spend trung bình cao nhất
    avg_spends = {c: df_temp[df_temp['Cluster'] == c]['Total_Spend'].mean() for c in cluster_ids}
    vip_cluster = max(avg_spends, key=avg_spends.get)
    profiles[vip_cluster] = "Cụm VIP / Cao cấp: Tổng chi tiêu vượt trội ở tất cả các ngành hàng"

    # 2. Xử lý các cụm còn lại
    remaining = [c for c in cluster_ids if c != vip_cluster]
    if len(remaining) >= 2:
        horeca_ratios = {}
        for c in remaining:
            sub = df_temp[df_temp['Cluster'] == c]
            fresh_frozen = (sub['Fresh'] + sub['Frozen']).sum()
            tot = sub['Total_Spend'].sum() + 1e-6
            horeca_ratios[c] = fresh_frozen / tot

        horeca_cluster = max(horeca_ratios, key=horeca_ratios.get)
        profiles[horeca_cluster] = "Cụm Nhà hàng / Khách sạn (HoReCa): Nhu cầu Thực phẩm tươi sống & Đông lạnh cao"

        retail_cluster = [c for c in remaining if c != horeca_cluster][0]
        profiles[retail_cluster] = "Cụm Bán lẻ phổ thông (Retail): Nhu cầu Tạp hóa, Sữa & Chất tẩy rửa cao"
    else:
        for c in remaining:
            profiles[c] = f"Cụm {c}: Phân khúc tiêu dùng phổ thông"

    return profiles

