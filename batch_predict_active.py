
import sys
import subprocess
from pathlib import Path


FACES_ACTIVE   = Path(r"D:\Data Science Research Project\naturalsocialtouch_datascience\main_faces_active").resolve()
FEATURES_ACTIVE= Path(r"D:\Data Science Research Project\naturalsocialtouch_datascience\features_active").resolve()
EMO_SCRIPT     = Path(__file__).parent / "batch_predict_emonet.py"
PYTHON         = sys.executable

def main():
    for video_dir in sorted(FACES_ACTIVE.iterdir()):
        if not video_dir.is_dir(): continue

        # 每个视频可能有多个 track 子文件夹
        for track_dir in sorted(video_dir.glob("track_*")):
            out_dir = FEATURES_ACTIVE / video_dir.name / track_dir.name
            if out_dir.exists() and (out_dir/"emonet.json").exists():
                print(f"✅ Skip {video_dir.name}/{track_dir.name} (already has emonet.json)")
                continue

            out_dir.mkdir(parents=True, exist_ok=True)
            cmd = [
                PYTHON,
                str(EMO_SCRIPT),
                str(track_dir),
                str(out_dir)
            ]
            print(f"\n>> Predicting EmoNet for {video_dir.name}/{track_dir.name}")
            # 调用情感预测脚本
            res = subprocess.run(cmd, check=False, capture_output=True, text=True)
            if res.returncode != 0:
                print(f"  [ERROR] failed: {res.stderr.strip() or res.stdout.strip()}")
            else:
                print(res.stdout.strip())

    print(f"\n All done! Check results under {FEATURES_ACTIVE}")

if __name__ == "__main__":
    main()
