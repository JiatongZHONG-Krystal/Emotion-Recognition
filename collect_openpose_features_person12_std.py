# 从 OpenPose JSON 中，为每个视频生成 person_1 / person_2 的 BODY_25 关键点
#       的均值与标准差特征；用“面积Top-2 + 最近中心匹配”在跨帧保持身份稳定；
#       过滤低质量（frame_count_pose < 10），并同时输出长表与宽表。


import json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# ====== 路径与配置 ======
OPENPOSE_OUTPUT_DIR = Path(r"openpose/output/processed")
OUTFILE_LONG = Path(r"openpose/output/openpose_features_person12_std.csv")
OUTFILE_WIDE = Path(r"openpose/output/openpose_features_person12_std_wide.csv")

NUM_KEYPOINTS = 25  # BODY_25
FEAT_MEAN = [f"pose_x{i+1}" for i in range(NUM_KEYPOINTS)] + \
            [f"pose_y{i+1}" for i in range(NUM_KEYPOINTS)]
FEAT_STD  = [f"pose_x_std{i+1}" for i in range(NUM_KEYPOINTS)] + \
            [f"pose_y_std{i+1}" for i in range(NUM_KEYPOINTS)]

MIN_FRAMES = 10  # 质量过滤阈值：轨迹参与帧数 < 10 则丢弃


def read_keypoints_from_json(json_path: Path) -> np.ndarray:
    """读取单帧 JSON，返回 (n_person, 25, 2) 坐标（无置信度，缺失为NaN；无人返回 shape=(0,25,2)）"""
    with open(json_path, "r") as f:
        data = json.load(f)
    people = data.get("people", [])
    if not people:
        return np.full((0, NUM_KEYPOINTS, 2), np.nan, dtype=float)
    all_kps = []
    for p in people:
        arr = np.array(p.get("pose_keypoints_2d", []), dtype=float)
        if arr.size == 0:
            continue
        arr = arr.reshape(-1, 3)[:, :2]            # 只要 (x,y)
        arr = arr[:NUM_KEYPOINTS]                  # 取前25个点
        if arr.shape[0] < NUM_KEYPOINTS:          # 不足补NaN
            pad = np.full((NUM_KEYPOINTS - arr.shape[0], 2), np.nan)
            arr = np.vstack([arr, pad])
        all_kps.append(arr)
    if not all_kps:
        return np.full((0, NUM_KEYPOINTS, 2), np.nan, dtype=float)
    return np.stack(all_kps, axis=0)

def _safe_center(kps: np.ndarray) -> np.ndarray:
    #人体中心：优先颈(1)与髋(8)中点；缺失退化为全点均值
    neck, midhip = kps[1], kps[8]
    if np.isnan(neck).any() or np.isnan(midhip).any():
        return np.nanmean(kps, axis=0)
    return (neck + midhip) / 2.0

def _bbox_area(kps: np.ndarray) -> float:
    #关键点包围盒面积，用于挑 Top-2 主体#
    xs, ys = kps[:, 0], kps[:, 1]
    valid = (~np.isnan(xs)) & (~np.isnan(ys))
    if valid.sum() < 4:
        return 0.0
    x1, x2 = float(np.nanmin(xs[valid])), float(np.nanmax(xs[valid]))
    y1, y2 = float(np.nanmin(ys[valid])), float(np.nanmax(ys[valid]))
    return max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))

def process_video_stable_tracks(json_dir: Path):
    json_files = sorted(json_dir.glob("*.json"))
    if not json_files:
        return []

    tracks = {0: [], 1: []}
    last_centers = {0: None, 1: None}

    for jf in json_files:
        kps_all = read_keypoints_from_json(jf)
        if kps_all.shape[0] == 0:
            continue

        # 1) 每帧挑面积最大的前两人
        scored = []
        for kps in kps_all:
            scored.append((_bbox_area(kps), _safe_center(kps), kps))
        scored.sort(key=lambda t: t[0], reverse=True)
        scored = scored[:2]

        # 2) 最近中心匹配：将本帧的1~2人分配到两条轨迹
        candidates = []
        for tid in (0, 1):
            lc = last_centers[tid]
            for j, (_, c, kps) in enumerate(scored):
                if lc is None or np.isnan(lc).any() or np.isnan(c).any():
                    dist = np.inf if lc is None else np.linalg.norm(lc - c)
                else:
                    dist = np.linalg.norm(lc - c)
                candidates.append((tid, j, dist))

        taken_t, taken_p = set(), set()
        for tid, j, _ in sorted(candidates, key=lambda x: x[2]):
            if tid in taken_t or j in taken_p:
                continue
            kps = scored[j][2]
            tracks[tid].append(kps)
            last_centers[tid] = _safe_center(kps)
            taken_t.add(tid); taken_p.add(j)
        # 若只匹配到一个人，另一条轨迹本帧留空

    rows = []
    for tid in (0, 1):
        if len(tracks[tid]) == 0:
            mean_kps = np.full((NUM_KEYPOINTS, 2), np.nan)
            std_kps  = np.full((NUM_KEYPOINTS, 2), np.nan)
            fc = 0
        else:
            arr = np.stack(tracks[tid], axis=0)
            mean_kps = np.nanmean(arr, axis=0)
            std_kps  = np.nanstd(arr, axis=0)
            fc = arr.shape[0]

        feat_mean = np.concatenate([mean_kps[:, 0], mean_kps[:, 1]])
        feat_std  = np.concatenate([std_kps[:, 0],  std_kps[:, 1]])

        rows.append({
            "person_temp": f"track_{tid+1}",
            "frame_count_pose": int(fc),
            **{c: (float(v) if np.isfinite(v) else np.nan) for c, v in zip(FEAT_MEAN, feat_mean)},
            **{c: (float(v) if np.isfinite(v) else np.nan) for c, v in zip(FEAT_STD,  feat_std)},
        })
    return rows

def main():
    results = []
    video_dirs = sorted([p for p in OPENPOSE_OUTPUT_DIR.glob("*") if p.is_dir()])

    for vf in tqdm(video_dirs, desc="Processing videos"):
        json_dir = vf / "json"
        if not json_dir.exists():
            continue

        rows = process_video_stable_tracks(json_dir)
        if not rows:
            continue

        # 质量过滤：丢掉 frame_count_pose < MIN_FRAMES 的轨迹
        rows = [r for r in rows if r["frame_count_pose"] >= MIN_FRAMES]
        if not rows:
            continue

        # 统一身份：按 frame_count_pose 降序 → person_1 / person_2
        rows_sorted = sorted(rows, key=lambda r: r["frame_count_pose"], reverse=True)
        for i, r in enumerate(rows_sorted[:2]):   # 最多两人
            r["video"] = vf.name
            r["person_label"] = f"person_{i+1}"
            results.append(r)

    # —— 长表（long）——
    df_long = pd.DataFrame(results)
    if df_long.empty:
        OUTFILE_LONG.parent.mkdir(parents=True, exist_ok=True)
        df_long.to_csv(OUTFILE_LONG, index=False, encoding="utf-8")
        print(f"[WARN] No valid rows. Saved empty long file to: {OUTFILE_LONG}")
        # 同步生成空宽表
        OUTFILE_WIDE.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=["video"]).to_csv(OUTFILE_WIDE, index=False, encoding="utf-8")
        print(f"[WARN] Saved empty wide file to: {OUTFILE_WIDE}")
        return

    # 标记 has_both（同一视频是否同时有 person_1 与 person_2）
    has_both = df_long.groupby("video")["person_label"].nunique().reset_index()
    has_both["has_both"] = has_both["person_label"] == 2
    df_long = df_long.merge(has_both[["video", "has_both"]], on="video", how="left")

    OUTFILE_LONG.parent.mkdir(parents=True, exist_ok=True)
    df_long.to_csv(OUTFILE_LONG, index=False, encoding="utf-8")
    print(f"Saved long format ({len(df_long)} rows) to {OUTFILE_LONG}")

    # —— 宽表（wide）：将 person_1 / person_2 展开到同一行 ——
    df_wide = df_long.pivot(index="video", columns="person_label")

    # 扁平化多级列名：('pose_x1', 'person_1') → 'pose_x1_1'
    df_wide.columns = [f"{col}_{plabel[-1]}" for col, plabel in df_wide.columns]
    df_wide = df_wide.reset_index()

    # 宽表里也保留 has_both（从任意一列还原）
    if "has_both_1" in df_wide.columns:
        df_wide = df_wide.rename(columns={"has_both_1": "has_both"})
        if "has_both_2" in df_wide.columns:
            df_wide = df_wide.drop(columns=["has_both_2"])
    elif "has_both_2" in df_wide.columns:
        df_wide = df_wide.rename(columns={"has_both_2": "has_both"})

    df_wide.to_csv(OUTFILE_WIDE, index=False, encoding="utf-8")
    print(f"Saved wide format ({len(df_wide)} rows) to {OUTFILE_WIDE}")

if __name__ == "__main__":
    main()
