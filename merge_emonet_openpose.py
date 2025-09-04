# merge_emonet_openpose.py
from pathlib import Path
import pandas as pd

BASE = Path(r"D:\Data Science Research Project\tables")
EMO_IN  = BASE / "features_with_labels.xlsx"                 # EmoNet + labels（长表）
POSE_IN = BASE / "openpose_features_person12_std_wide.xlsx"  # OpenPose（宽表）

OUT_ALL  = BASE / "merged_features.csv"                      # 合并后全量
OUT_BOTH = BASE / "merged_features_has_both.csv"             # 仅双人均有效

MIN_FRAMES_EMO = 10

def main():
    # 读取
    emo  = pd.read_excel(EMO_IN)
    pose = pd.read_excel(POSE_IN)

    # 统一 video 为字符串 & 去空白,避免因为空格或类型不同导致合并不上。
    emo["video"]  = emo["video"].astype(str).str.strip()
    pose["video"] = pose["video"].astype(str).str.strip()

    # EmoNet 长表、选 Top-2 轨迹、宽表
    emo["frame_count"] = pd.to_numeric(emo["frame_count"], errors="coerce")
    emo_f = emo[emo["frame_count"].fillna(0) >= MIN_FRAMES_EMO].copy()

    # 每视频按 frame_count 排序，取前2条
    emo_f["rank"] = emo_f.groupby("video")["frame_count"].rank(method="first", ascending=False)
    emo_top2 = emo_f[emo_f["rank"] <= 2].copy()
    emo_top2["person_label"] = emo_top2["rank"].map({1.0: "1", 2.0: "2"})

    # 选要宽表化的特征（按需增减）
    emo_feat_cols = ["valence", "arousal", "frame_count"]
    emo_wide = emo_top2.pivot_table(index="video",
                                    columns="person_label",
                                    values=emo_feat_cols,
                                    aggfunc="first")
    emo_wide.columns = [f"{v}_{p}" for v, p in emo_wide.columns]
    emo_wide = emo_wide.reset_index()

    # 拼回视频级标签
    label_cols = [
        'touch valence (1:pleasant, 0:neutral, -1:unpleasant)',
        'touch action (such as hug, handshake, slap, stroke, tap, pat, push, punch and so on)',
        'touch giver, valence  (1:pleasant, 0:neutral, -1:unpleasant)',
        'touch receiver, valence  (1:pleasant, 0:neutral, -1:unpleasant)',
    ]
    # 只合并存在的标签列（防止列名有变化）
    keep_labels = ["video"] + [c for c in label_cols if c in emo.columns]
    emo_labels = emo.drop_duplicates("video")[keep_labels]
    emo_wide = emo_wide.merge(emo_labels, on="video", how="left")

    # EmoNet 侧 has_both（两人都 >= MIN_FRAMES_EMO）
    emo_wide["has_both_emo"] = (emo_wide.get("frame_count_1", 0).fillna(0) >= MIN_FRAMES_EMO) & \
                               (emo_wide.get("frame_count_2", 0).fillna(0) >= MIN_FRAMES_EMO)

    # === 与 OpenPose合并 ===
    merged = pd.merge(emo_wide, pose, on="video", how="inner")

    # 两边都双人有效
    if "has_both" in merged.columns:
        merged["has_both_both"] = merged["has_both_emo"].fillna(False) & merged["has_both"].fillna(False)
    else:
        merged["has_both_both"] = merged["has_both_emo"].fillna(False)

    # 保存
    OUT_ALL.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUT_ALL, index=False, encoding="utf-8-sig")
    print(f"[OK] merged saved: {OUT_ALL}  (rows={len(merged)})")

    merged_has_both = merged[merged["has_both_both"]].copy()
    merged_has_both.to_csv(OUT_BOTH, index=False, encoding="utf-8-sig")
    print(f"[OK] merged (has_both) saved: {OUT_BOTH}  (rows={len(merged_has_both)})")

if __name__ == "__main__":
    main()
