"""
preprocess.py — Data preprocessing pipeline cho ET-SSL.

Chức năng:
1. Đọc dataset CSV (UNSW-NB15 hoặc CIC-Darknet2020)
2. Map cột về 20 feature chuẩn của model
3. Làm sạch: xử lý NaN/Inf, clip outlier
4. Tạo/lưu scaler mới (StandardScaler)
5. Chia train/val/test = 70/15/15
6. Lưu processed data vào data/processed/

Output:
  data/processed/{dataset}/
    ├── X_train.npy, y_train.npy
    ├── X_val.npy,   y_val.npy
    ├── X_test.npy,  y_test.npy
    └── scaler.pkl

Usage:
    python data/preprocess.py --dataset unsw_nb15 --data_path data/raw/unsw-nb15/
    python data/preprocess.py --dataset cic_darknet2020 --data_path data/raw/cic-darknet2020/
    python data/preprocess.py --generate_synthetic --n_samples 10000   # Test mode
"""

import argparse
import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.feature_schema import (
    MODEL_FEATURES,
    map_columns,
    get_schema,
    detect_label_column,
)

PROCESSED_DIR = Path(__file__).parent / "processed"


# =====================================================================
# 1. Load Dataset
# =====================================================================
def load_dataset_csv(data_path: str | Path, dataset_name: str) -> pd.DataFrame:
    """
    Đọc dataset CSV. Hỗ trợ single file hoặc thư mục chứa nhiều CSV.

    Args:
        data_path: Đường dẫn file CSV hoặc thư mục
        dataset_name: tên dataset để log

    Returns:
        DataFrame gộp tất cả CSV trong thư mục (hoặc file đơn)
    """
    data_path = Path(data_path)

    if data_path.is_file():
        print(f"📂 Loading single file: {data_path.name}")
        df = pd.read_csv(data_path, low_memory=False)
    elif data_path.is_dir():
        csv_files = sorted(data_path.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files in: {data_path}")
        print(f"📂 Loading {len(csv_files)} CSV files from: {data_path}")
        dfs = []
        for f in csv_files:
            print(f"  Reading {f.name}...")
            dfs.append(pd.read_csv(f, low_memory=False))
        df = pd.concat(dfs, ignore_index=True)
    else:
        raise FileNotFoundError(f"Path not found: {data_path}")

    print(f"  ✅ Loaded: {len(df):,} rows × {len(df.columns)} columns")
    return df


# =====================================================================
# 2. Làm sạch dữ liệu
# =====================================================================
def clean_features(X: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Làm sạch feature matrix:
    - Thay NaN bằng median của cột
    - Thay Inf bằng max hợp lệ của cột
    - Clip outlier tại ±5 IQR
    """
    X = X.copy()

    # Xử lý Inf
    n_inf = np.isinf(X.values).sum()
    if n_inf > 0:
        if verbose:
            print(f"  ⚠️  {n_inf} Inf values → replaced with col max")
        for col in X.columns:
            mask_inf = np.isinf(X[col])
            if mask_inf.any():
                finite_max = X.loc[~mask_inf, col].max()
                X.loc[mask_inf, col] = finite_max

    # Xử lý NaN
    n_nan = X.isna().sum().sum()
    if n_nan > 0:
        if verbose:
            print(f"  ⚠️  {n_nan} NaN values → replaced with col median")
        for col in X.columns:
            if X[col].isna().any():
                X[col].fillna(X[col].median(), inplace=True)

    # Clip outlier tại ±5 IQR
    for col in X.columns:
        q1 = X[col].quantile(0.25)
        q3 = X[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 5 * iqr
        upper = q3 + 5 * iqr
        n_clipped = ((X[col] < lower) | (X[col] > upper)).sum()
        if n_clipped > 0:
            X[col] = X[col].clip(lower, upper)

    if verbose:
        print(f"  ✅ Cleaned: no NaN/Inf remaining")

    return X


# =====================================================================
# 3. Extract label
# =====================================================================
def extract_labels(df: pd.DataFrame, dataset_name: str) -> np.ndarray:
    """
    Trích nhãn binary (0=normal, 1=attack).

    Returns:
        y: (N,) int array — 0=normal, 1=anomaly
    """
    schema = get_schema(dataset_name)
    label_col = detect_label_column(df, dataset_name)

    if label_col is None:
        raise ValueError(
            f"Cannot find label column in DataFrame. "
            f"Available columns: {list(df.columns[:10])}..."
        )

    print(f"  Label column: '{label_col}'")
    raw_labels = df[label_col]

    # Encode về binary
    normal_val = schema["normal_val"]
    if isinstance(normal_val, str):
        y = (raw_labels != normal_val).astype(int).values
    else:
        y = (raw_labels != 0).astype(int).values

    n_normal = (y == 0).sum()
    n_attack = (y == 1).sum()
    print(f"  Normal: {n_normal:,} ({n_normal/len(y)*100:.1f}%) | Attack: {n_attack:,} ({n_attack/len(y)*100:.1f}%)")
    return y


# =====================================================================
# 4. Split & Save
# =====================================================================
def split_and_save(
    X: np.ndarray,
    y: np.ndarray,
    dataset_name: str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    random_seed: int = 42,
    save_scaler_to_model: bool = True,
) -> dict:
    """
    Chia train/val/test, fit scaler trên train, lưu files.

    ET-SSL là self-supervised: chỉ dùng normal data để train.
    Tuy nhiên ta vẫn lưu toàn bộ data để eval.

    Returns:
        dict với đường dẫn các file đã lưu
    """
    out_dir = PROCESSED_DIR / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Split stratified
    test_ratio = 1.0 - train_ratio - val_ratio
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=(val_ratio + test_ratio), random_state=random_seed, stratify=y
    )
    val_size_rel = val_ratio / (val_ratio + test_ratio)
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=(1 - val_size_rel), random_state=random_seed, stratify=y_tmp
    )

    print(f"\n📊 Split:")
    print(f"  Train: {len(X_train):,} (normal={sum(y_train==0):,})")
    print(f"  Val:   {len(X_val):,} (normal={sum(y_val==0):,})")
    print(f"  Test:  {len(X_test):,} (normal={sum(y_test==0):,})")

    # Fit scaler trên NORMAL train data (như ET-SSL paper)
    X_train_normal = X_train[y_train == 0]
    print(f"\n🔧 Fitting StandardScaler on {len(X_train_normal):,} normal train samples...")
    scaler = StandardScaler()
    scaler.fit(X_train_normal)

    # Transform
    X_train_s = scaler.transform(X_train).astype(np.float32)
    X_val_s = scaler.transform(X_val).astype(np.float32)
    X_test_s = scaler.transform(X_test).astype(np.float32)

    # Lưu
    np.save(out_dir / "X_train.npy", X_train_s)
    np.save(out_dir / "y_train.npy", y_train)
    np.save(out_dir / "X_val.npy", X_val_s)
    np.save(out_dir / "y_val.npy", y_val)
    np.save(out_dir / "X_test.npy", X_test_s)
    np.save(out_dir / "y_test.npy", y_test)

    scaler_path = out_dir / "scaler.pkl"
    joblib.dump(scaler, scaler_path)
    print(f"  ✅ Saved scaler: {scaler_path}")

    # Copy scaler vào model dir để inference dùng được
    if save_scaler_to_model:
        model_dir = Path(__file__).parent.parent / \
            "TrafficGuard/models/edge_ai-20260716T101644Z-1-001/edge_ai"
        if model_dir.exists():
            import shutil
            dest = model_dir / "scaler.pkl"
            shutil.copy2(scaler_path, dest)
            print(f"  ✅ Copied scaler to model dir: {dest}")

    return {
        "out_dir": str(out_dir),
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_test": len(X_test),
        "scaler_path": str(scaler_path),
    }


# =====================================================================
# 5. Synthetic data generator (test mode)
# =====================================================================
def generate_synthetic(
    n_samples: int = 10000,
    n_features: int = 20,
    anomaly_ratio: float = 0.2,
    random_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Tạo synthetic data giả lập traffic network:
    - Normal: Gaussian(mean=0, std=1)
    - Anomaly: Gaussian(mean=3, std=1.5) — dịch chuyển phân phối
    """
    rng = np.random.default_rng(random_seed)
    n_attack = int(n_samples * anomaly_ratio)
    n_normal = n_samples - n_attack

    X_normal = rng.normal(0.0, 1.0, (n_normal, n_features)).astype(np.float32)
    X_attack = rng.normal(3.0, 1.5, (n_attack, n_features)).astype(np.float32)

    X = np.vstack([X_normal, X_attack])
    y = np.array([0] * n_normal + [1] * n_attack, dtype=np.int32)

    # Shuffle
    idx = rng.permutation(len(X))
    return X[idx], y[idx]


# =====================================================================
# Main
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="Preprocess traffic dataset for ET-SSL")
    parser.add_argument("--dataset", choices=["unsw_nb15", "cic_darknet2020"],
                        help="Dataset name")
    parser.add_argument("--data_path", help="Path to CSV file or directory")
    parser.add_argument("--generate_synthetic", action="store_true",
                        help="Generate synthetic data for testing")
    parser.add_argument("--n_samples", type=int, default=10000,
                        help="Samples for synthetic mode")
    parser.add_argument("--train_ratio", type=float, default=0.70)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    print("=" * 60)
    print("🔄 ET-SSL Data Preprocessing")
    print("=" * 60)

    if args.generate_synthetic:
        print(f"\n🎲 Generating synthetic data ({args.n_samples:,} samples)...")
        X, y = generate_synthetic(args.n_samples)
        print(f"  Shape: {X.shape} | Normal: {sum(y==0):,} | Attack: {sum(y==1):,}")
        result = split_and_save(
            X, y, "synthetic",
            args.train_ratio, args.val_ratio, args.seed
        )
    else:
        if not args.dataset or not args.data_path:
            parser.error("--dataset và --data_path bắt buộc khi không dùng --generate_synthetic")

        print(f"\n📦 Dataset: {args.dataset}")
        df = load_dataset_csv(args.data_path, args.dataset)

        print(f"\n🔗 Mapping columns to model features...")
        X_df = map_columns(df, args.dataset, verbose=True)

        print(f"\n🔍 Extracting labels...")
        y = extract_labels(df, args.dataset)

        print(f"\n🧹 Cleaning features...")
        X_df = clean_features(X_df, verbose=True)
        X = X_df.values.astype(np.float32)

        result = split_and_save(
            X, y, args.dataset,
            args.train_ratio, args.val_ratio, args.seed
        )

    print("\n" + "=" * 60)
    print("✅ Preprocessing complete!")
    print(f"  Output: {result['out_dir']}")
    print(f"  Train/Val/Test: {result['n_train']}/{result['n_val']}/{result['n_test']}")
    print(f"  Scaler: {result['scaler_path']}")


if __name__ == "__main__":
    main()
