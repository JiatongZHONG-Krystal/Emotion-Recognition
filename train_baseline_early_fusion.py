# 用早期融合（EmoNet + OpenPose 特征拼接）训练 MLP，多折交叉验证评估

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
import joblib

BASE = Path(r"D:\Data Science Research Project\tables")
INFILE = BASE / "merged_features_has_both.csv"
OUTDIR = BASE / "baseline_results_cv"
OUTDIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INFILE)

# 找到标签列
label_candidates = [c for c in df.columns if "touch valence (1:pleasant" in c]
if not label_candidates:
    raise ValueError("未找到 'touch valence' 标签列，请确认列名。")
LABEL_COL = label_candidates[0]

# 丢掉非数值列
drop_cols = {"video", "has_both", "has_both_emo", "has_both_both"}
drop_cols |= set(label_candidates)
drop_cols |= {c for c in df.columns if "touch action " in c.lower()}

# 特征
num_cols = [c for c in df.columns if c not in drop_cols]
X = df[num_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).values

# 标签
y_raw = df[LABEL_COL].astype(int).values
classes_sorted = np.sort(np.unique(y_raw))
cls_to_idx = {c: i for i, c in enumerate(classes_sorted)}
idx_to_cls = {i: c for c, i in cls_to_idx.items()}
y = np.array([cls_to_idx[v] for v in y_raw])

# 模型
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

# 交叉验证
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 用于存储每个fold的混淆矩阵
all_cm = []

# 循环交叉验证
for train_idx, val_idx in cv.split(X, y):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    # 训练模型
    pipe.fit(X_train, y_train)

    # 预测
    pred = pipe.predict(X_val)

    # 计算混淆矩阵
    cm = confusion_matrix(y_val, pred)
    all_cm.append(cm)

# 计算每个fold的混淆矩阵并进行汇总
mean_cm = np.mean(all_cm, axis=0)

# 评估
accs = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")
f1s = cross_val_score(pipe, X, y, cv=cv, scoring="f1_macro")

print("\n=== Baseline (Early Fusion + MLP, 5-fold CV) ===")
print(f"Accuracy: {accs.mean():.4f} ± {accs.std():.4f}")
print(f"Macro-F1: {f1s.mean():.4f} ± {f1s.std():.4f}")

# 输出混淆矩阵
print("\nConfusion Matrix (Average over all folds):")
print(mean_cm)

# 保存结果
pd.DataFrame({
    "metric": ["accuracy_mean", "accuracy_std", "macro_f1_mean", "macro_f1_std"],
    "value": [accs.mean(), accs.std(), f1s.mean(), f1s.std()]
}).to_csv(OUTDIR / "metrics_cv.csv", index=False, encoding="utf-8-sig")

# 保存混淆矩阵
pd.DataFrame(mean_cm).to_csv(OUTDIR / "confusion_matrix_cv.csv", index=False, encoding="utf-8-sig")

# 保存模型
joblib.dump({"pipeline": pipe, "feature_cols": num_cols, "label_map": idx_to_cls},
            OUTDIR / "baseline_mlp_cv.joblib")

print(f"\n[OK] Saved metrics → {OUTDIR / 'metrics_cv.csv'}")
print(f"[OK] Saved confusion matrix → {OUTDIR / 'confusion_matrix_cv.csv'}")
print(f"[OK] Saved model (pipeline structure only) → {OUTDIR / 'baseline_mlp_cv.joblib'}")
