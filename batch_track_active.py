import sys
import subprocess
from pathlib import Path


FRAMES_ROOT = Path(r"D:\Data Science Research Project\naturalsocialtouch_datascience\frames").resolve()
ACTIVE_ROOT = Path(r"D:\Data Science Research Project\naturalsocialtouch_datascience\main_faces_active").resolve()
TOP_K       = 2  # 每个视频选前 K 条最活跃轨迹

PYTHON       = sys.executable
TRACK_SCRIPT = Path(__file__).parent / "track_main_subject.py"

def main():
    for vid_folder in sorted(FRAMES_ROOT.iterdir()):
        if not vid_folder.is_dir():
            continue

        out_folder = ACTIVE_ROOT / vid_folder.name

        # —— 如果已经有裁剪结果，跳过 ——
        if out_folder.exists() and any(out_folder.glob("track_*")):
            print(f" Skip {vid_folder.name} (already done)")
            continue

        print(f"\n>> Processing {vid_folder.name}")
        cmd = [
            PYTHON, str(TRACK_SCRIPT),
            "--frames_dir", str(vid_folder),
            "--out_dir",    str(out_folder),
            "--top_k",      str(TOP_K)
        ]

        # 调用脚本，不抛异常，手动检查返回码
        result = subprocess.run(cmd, check=False, capture_output=True, text=True)
        if result.returncode != 0:
            msg = result.stdout.strip() or result.stderr.strip()
            print(f"  [WARN] skipped {vid_folder.name}: {msg}")
            continue
        # 脚本正常输出
        print(result.stdout.strip())

    print(f"\n All done! Check folders under {ACTIVE_ROOT}")

if __name__ == "__main__":
    main()
