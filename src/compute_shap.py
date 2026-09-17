"""
Compute SHAP feature importance for the XGBoost transfer_bytes model.

Uses a self-contained 8K-row sample from the raw CSV so no split alignment
is needed. Target encodings are computed from the same sample (slight in-sample
bias for those 2 features, but negligible for SHAP aggregate statistics).

Output: logs/shap_results.json
"""

import json
import sys
import numpy as np
import pandas as pd
import xgboost as xgb
import shap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "per_request" / "xgb_transfer_bytes.json"
DATA_PATH  = ROOT / "data" / "raw" / "per_request_1pct.csv"
EMBED_PATH = ROOT / "models" / "per_request" / "url_embedder.joblib"
LOGS = ROOT / "logs"
LOGS.mkdir(exist_ok=True)
OUT = LOGS / "shap_results.json"

sys.path.insert(0, str(Path(__file__).parent / "model"))
from train_multi_target import engineer_features

# Feature group mapping (matches booster.feature_names order)
GROUPS = {
    "Domain identity": ["domain_median_bytes", "domain_type_median"],
    "URL structure": [
        "path_depth", "url_length", "num_query_params", "has_query_params",
        "ext_js", "ext_gif", "ext_png", "ext_jpg", "ext_html",
        "ext_php", "ext_json", "ext_css",
    ],
    "URL content (TF-IDF)": [f"url_emb_{i}" for i in range(50)],
    "Request metadata": [
        "rt_script", "rt_image", "rt_other", "rt_html", "rt_text", "rt_css",
        "init_script", "init_parser", "init_other", "is_post",
    ],
    "URL patterns": [
        "path_has_js", "path_has_collect", "path_has_image",
        "path_has_sync", "path_has_ad", "path_has_api",
    ],
}

print("Loading data sample...")
df = pd.read_csv(DATA_PATH, low_memory=False)

# Use a stratified-ish sample: 8K rows spread across the dataset
np.random.seed(2024)
sample_idx = np.random.choice(len(df), size=8000, replace=False)
df_sample = df.iloc[sample_idx].reset_index(drop=True)
print(f"Sample size: {len(df_sample)}")

# Load saved URL embedder and compute embeddings
print("Computing URL embeddings...")
import joblib
embedder = joblib.load(str(EMBED_PATH))
url_embeddings = embedder.transform(df_sample["url_path"].fillna(""))
print(f"URL embeddings shape: {url_embeddings.shape}")

# Engineer features: use sample as its own train_df for target encoding
# (slight in-sample bias for domain_median_bytes / domain_type_median,
# but acceptable for SHAP aggregate importance)
print("Engineering features...")
X = engineer_features(df_sample, df_sample, url_embeddings=url_embeddings)
print(f"Feature matrix: {X.shape}, columns: {list(X.columns[:5])}...")

# Load booster
print("Loading model...")
booster = xgb.Booster()
booster.load_model(str(MODEL_PATH))
feat_names = booster.feature_names
print(f"Model feature count: {len(feat_names)}")

# Align feature matrix to booster's expected order
X_aligned = X[feat_names].values.astype(np.float32)
dmat = xgb.DMatrix(X_aligned, feature_names=feat_names)

# Compute SHAP values using XGBoost's native pred_contribs
print("Computing SHAP values...")
shap_values = booster.predict(dmat, pred_contribs=True)
# shap_values shape: (n_samples, n_features + 1) — last column is bias
shap_main = shap_values[:, :-1]  # drop bias term
mean_abs_shap = np.abs(shap_main).mean(axis=0)

# Build per-feature results
per_feature = {}
for name, val in zip(feat_names, mean_abs_shap):
    per_feature[name] = float(val)

# Build per-group results
total = sum(per_feature.values())
per_group = {}
for group, features in GROUPS.items():
    group_total = sum(per_feature.get(f, 0) for f in features)
    per_group[group] = {
        "mean_abs_shap": round(group_total, 4),
        "pct_of_total": round(group_total / total * 100, 2),
    }

# Top 15 individual features
top15 = sorted(per_feature.items(), key=lambda x: -x[1])[:15]

# Also compute gain-based importance for comparison
gain_scores = booster.get_score(importance_type="gain")
gain_total = sum(gain_scores.values())
gain_by_group = {}
for group, features in GROUPS.items():
    g = sum(gain_scores.get(f, 0) for f in features)
    gain_by_group[group] = round(g / gain_total * 100, 2) if gain_total else 0

results = {
    "per_group_shap": per_group,
    "per_group_gain": gain_by_group,
    "top15_features": [{"feature": k, "mean_abs_shap": round(v, 4)} for k, v in top15],
    "n_samples": len(df_sample),
    "total_mean_abs_shap": round(total, 4),
}

with open(OUT, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nSaved to {OUT}")
print("\nFeature group importance (SHAP vs Gain):")
print(f"{'Group':<30} {'SHAP%':>8} {'Gain%':>8}")
for group in GROUPS:
    s = per_group[group]["pct_of_total"]
    g = gain_by_group[group]
    print(f"  {group:<28} {s:>7.1f}% {g:>7.1f}%")

print("\nTop 15 features by mean |SHAP|:")
for feat, val in top15:
    print(f"  {feat:<35} {val:.4f}")
