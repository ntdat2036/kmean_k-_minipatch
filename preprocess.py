"""
preprocess.py
=============
Module tiền xử lý dữ liệu tập trung dùng chung duy nhất cho cả:
  - Huấn luyện mô hình (main.py)
  - Suy luận dự đoán dữ liệu mới (predict_new.py)

Bảo đảm 100% nhất quán đặc trưng, tên cột và tỷ lệ biến đổi.
"""

import numpy as np
import pandas as pd

FEATURE_COLS = ["Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicassen"]


def validate_input_data(df):
    """Kiểm tra hợp lệ dữ liệu đầu vào trước khi trích xuất đặc trưng."""
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Dữ liệu thiếu các cột chi tiêu bắt buộc: {missing}")

    for c in FEATURE_COLS:
        if not pd.api.types.is_numeric_dtype(df[c]):
            raise TypeError(f"Cột '{c}' phải là kiểu dữ liệu số (numeric).")

    if (df[FEATURE_COLS] < 0).any().any():
        raise ValueError("Chi tiêu ngành hàng không được chứa giá trị âm.")


def build_features(df):
    """
    Feature engineering DUY NHẤT — dùng cho cả train (main.py) và predict (predict_new.py).
    1. Kiểm tra tính hợp lệ dữ liệu
    2. Biến đổi log1p cho 6 ngành hàng gốc
    3. Biến đổi log1p cho 3 tỷ lệ chi tiêu (Fresh_Ratio_log, NonEssential_Ratio_log, Grocery_Milk_Ratio_log)
    """
    validate_input_data(df)
    df_proc = df.copy()

    for c in FEATURE_COLS:
        df_proc[f"{c}_log"] = np.log1p(df_proc[c])

    total_spend = df_proc[FEATURE_COLS].sum(axis=1)
    df_proc["Fresh_Ratio_log"]        = np.log1p(df_proc["Fresh"] / (total_spend + 1e-6))
    df_proc["NonEssential_Ratio_log"] = np.log1p((df_proc["Grocery"] + df_proc["Detergents_Paper"]) / (total_spend + 1e-6))
    df_proc["Grocery_Milk_Ratio_log"] = np.log1p(df_proc["Grocery"] / (df_proc["Milk"] + 1e-6))

    training_cols = [f"{c}_log" for c in FEATURE_COLS] + [
        "Fresh_Ratio_log", "NonEssential_Ratio_log", "Grocery_Milk_Ratio_log"
    ]
    return df_proc[training_cols]
