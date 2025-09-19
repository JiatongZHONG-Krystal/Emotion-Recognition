from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
import joblib

# ===== 路径 =====
BASE = Path(r"D:\Data Science Research Project\tables")
INFILE = BASE / "merged_features_has_both.csv"
OUTDIR = BASE / "baseline_results_single_modal"
OUTDIR.mkdir(parents=True, exist_ok=True)

# ===== 读取数据 =====
df = pd.read_csv(INFILE)

# 找到标签列
label_candidates = [c for c in df.columns if "touch valence" in c]
if not label_candidates:
    raise ValueError("未找到 'touch valence' 标签列，请确认列名。")
LABEL_COL = label_candidates[0]

# 丢掉非数值列
drop_cols = {"video", "has_both", "has_both_emo", "has_both_both"}
drop_cols |= set(label_candidates)
drop_cols |= {c for c in df.columns if "touch action " in c.lower()}

# 获取 EmoNet 和 OpenPose 特征列
emo_cols = ['valence_1', 'valence_2']
pose_cols = [c for c in df.columns if "pose" in c]

# 提取 EmoNet和OpenPose特征（用于训练单模态模型）
X_emo = df[emo_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).values
X_pose = df[pose_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).values

# 标签
y_raw = df[LABEL_COL].astype(int).values
classes_sorted = np.sort(np.unique(y_raw))
cls_to_idx = {c: i for i, c in enumerate(classes_sorted)}
idx_to_cls = {i: c for c, i in cls_to_idx.items()}
y = np.array([cls_to_idx[v] for v in y_raw])

# ===== 模型 =====
scaler = StandardScaler()
mlp = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    activation="relu",
    solver="adam",
    alpha=1e-4,
    batch_size=64,
    learning_rate_init=1e-3,
    max_iter=200,
    random_state=42,
    early_stopping=True,
    n_iter_no_change=10,
    verbose=False,
)
pipe = Pipeline([
    ("scaler", scaler),
    ("clf", mlp),
])

# ===== 交叉验证 =====
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 单模态 EmoNet 模型评估
accs_emo = cross_val_score(pipe, X_emo, y, cv=cv, scoring="accuracy")
f1s_emo = cross_val_score(pipe, X_emo, y, cv=cv, scoring="f1_macro")

# 单模态 OpenPose 模型评估
accs_pose = cross_val_score(pipe, X_pose, y, cv=cv, scoring="accuracy")
f1s_pose = cross_val_score(pipe, X_pose, y, cv=cv, scoring="f1_macro")

# 输出评估结果
print("\n=== Single Modality (EmoNet) ===")
print(f"Accuracy: {accs_emo.mean():.4f} ± {accs_emo.std():.4f}")
print(f"Macro-F1: {f1s_emo.mean():.4f} ± {f1s_emo.std():.4f}")

print("\n=== Single Modality (OpenPose) ===")
print(f"Accuracy: {accs_pose.mean():.4f} ± {accs_pose.std():.4f}")
print(f"Macro-F1: {f1s_pose.mean():.4f} ± {f1s_pose.std():.4f}")

# ===== 保存结果 =====
pd.DataFrame({
    "metric": ["accuracy_mean", "accuracy_std", "macro_f1_mean", "macro_f1_std"],
    "value": [accs_emo.mean(), accs_emo.std(), f1s_emo.mean(), f1s_emo.std()]
}).to_csv(OUTDIR/"metrics_emo.csv", index=False, encoding="utf-8-sig")

pd.DataFrame({
    "metric": ["accuracy_mean", "accuracy_std", "macro_f1_mean", "macro_f1_std"],
    "value": [accs_pose.mean(), accs_pose.std(), f1s_pose.mean(), f1s_pose.std()]
}).to_csv(OUTDIR/"metrics_pose.csv", index=False, encoding="utf-8-sig")

joblib.dump({"pipeline": pipe, "feature_cols": emo_cols, "label_map": idx_to_cls}, OUTDIR/"baseline_mlp_emo.joblib")
joblib.dump({"pipeline": pipe, "feature_cols": pose_cols, "label_map": idx_to_cls}, OUTDIR/"baseline_mlp_pose.joblib")

print(f"\n[OK] Saved metrics → {OUTDIR/'metrics_emo.csv'}")
print(f"[OK] Saved metrics → {OUTDIR/'metrics_pose.csv'}")
print(f"[OK] Saved model (EmoNet only) → {OUTDIR/'baseline_mlp_emo.joblib'}")
print(f"[OK] Saved model (OpenPose only) → {OUTDIR/'baseline_mlp_pose.joblib'}")
