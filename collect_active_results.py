import json
from pathlib import Path
import pandas as pd


FEATURES_ROOT = Path(r"D:\Data Science Research Project\naturalsocialtouch_datascience\features_active")
OUT_CSV = FEATURES_ROOT / "all_emonet_active.csv"

def collect_active_results(root: Path, out_csv: Path):
    rows = []
    for video_dir in sorted(root.iterdir()):
        if not video_dir.is_dir():
            continue
        for track_dir in sorted(video_dir.iterdir()):
            json_path = track_dir / "emonet.json"
            if not json_path.exists():
                continue
            # 读取单轨迹的情绪结果
            data = json.load(open(json_path, "r", encoding="utf-8"))
            # 添加元信息：视频名 + 轨迹名
            data["video"] = video_dir.name
            data["track"] = track_dir.name
            rows.append(data)

    if not rows:
        print("没找到任何 emonet.json，确认 FEATURES_ROOT 路径是否正确？")
        return

    # 构造 DataFrame
    df = pd.DataFrame(rows)
    # 只保留我们需要的列，并设置顺序
    df = df[['video', 'track', 'frame_count', 'valence', 'arousal']]
    # 保存为 CSV
    df.to_csv(out_csv, index=False, encoding='utf-8-sig')
    print(f"已保存 {len(df)} 行 → {out_csv}")

if __name__ == "__main__":
    collect_active_results(FEATURES_ROOT, OUT_CSV)
