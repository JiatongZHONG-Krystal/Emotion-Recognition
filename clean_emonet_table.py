# clean_emonet_table.py
from pathlib import Path
import pandas as pd
import numpy as np

INFILE  = Path(r"D:\Data Science Research Project\naturalsocialtouch_datascience\features_active\all_emonet_active.xlsx")
OUTFILE = Path(r"D:\Data Science Research Project\naturalsocialtouch_datascience\features_active\all_emonet_active_clean.csv")
MIN_FRAMES = 10   # 丢掉帧数 < 10 的轨迹
TOP_N      = 2    # 每个视频只保留帧数最多的 2 条轨迹
DROP_NA_VA = False  # 是否去掉 valence/arousal 缺失的行

def read_table(path: Path) -> pd.DataFrame:
    ext = path.suffix.lower()
    if ext == ".csv":
        return pd.read_csv(path)
    elif ext in (".xlsx", ".xls"):
        return pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

def main():
    if not INFILE.exists():
        raise FileNotFoundError(f"Input file not found: {INFILE}")

    df = read_table(INFILE)

    # 检查必需列
    required = {"video", "track", "frame_count"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # 类型转换
    df["frame_count"] = pd.to_numeric(df["frame_count"], errors="coerce")
    for col in ("valence", "arousal"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    n0, vids0 = len(df), df["video"].nunique()

    # 过滤低帧数和可选的 NaN
    df = df[df["frame_count"].fillna(0) >= MIN_FRAMES]
    if DROP_NA_VA and {"valence","arousal"} <= set(df.columns):
        df = df[~df["valence"].isna() & ~df["arousal"].isna()]
    n1, vids1 = len(df), df["video"].nunique()

    # 每个视频保留 TOP_N 条轨迹
    def sort_key(sub):
        if {"valence","arousal"} <= set(sub.columns):
            na = sub[["valence","arousal"]].isna().sum(axis=1)
        else:
            na = pd.Series(0, index=sub.index)
        key = pd.DataFrame({
            "frame_count": sub["frame_count"],
            "na": na,
            "track": sub["track"].astype(str)
        }, index=sub.index)
        return key.sort_values(by=["frame_count","na","track"], ascending=[False, True, True]).index

    picked_idx = []
    for vid, sub in df.groupby("video", sort=False):
        order = sort_key(sub)
        picked_idx.extend(list(order[:max(1, TOP_N)]))

    df_top = df.loc[picked_idx].copy()
    df_top = df_top.sort_values(["video","frame_count"], ascending=[True, False]).reset_index(drop=True)
    n2, vids2 = len(df_top), df_top["video"].nunique()

    # 保存结果
    OUTFILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUTFILE.with_suffix(OUTFILE.suffix + ".tmp")
    df_top.to_csv(tmp, index=False, encoding="utf-8")
    tmp.replace(OUTFILE)

    # 打印统计
    def _fmt(x): return f"{x:,}"
    print("=== Clean Summary ===")
    print(f"Input rows:      {_fmt(n0)}  | videos: {_fmt(vids0)}")
    print(f"After filters:   {_fmt(n1)}  | videos: {_fmt(vids1)}  (min_frames >= {MIN_FRAMES}"
          + (", drop_na_va" if DROP_NA_VA else "") + ")")
    print(f"After top-{TOP_N}: {_fmt(n2)}  | videos: {_fmt(vids2)}")
    if {"valence","arousal"} <= set(df_top.columns):
        print("Valence stats (mean±sd, min..max): "
              f"{df_top['valence'].mean():.3f}±{df_top['valence'].std():.3f}, "
              f"{df_top['valence'].min():.3f}..{df_top['valence'].max():.3f}")
        print("Arousal stats  (mean±sd, min..max): "
              f"{df_top['arousal'].mean():.3f}±{df_top['arousal'].std():.3f}, "
              f"{df_top['arousal'].min():.3f}..{df_top['arousal'].max():.3f}")
    print(f"Saved: {OUTFILE}")

if __name__ == "__main__":
    main()
