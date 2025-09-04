
import argparse
import subprocess
import sys
from pathlib import Path

def process_video(video, frames_root, faces_root, features_root, script_dir):
    stem = video.stem
    frames_dir = frames_root / stem

    # 1) 检查帧目录是否存在
    if not (frames_dir.exists() and any(frames_dir.glob("*.jpg"))):
        print(f"[WARN] No frames found for {stem} in {frames_dir}, skipping")
        return

    # 2) 裁剪主角人脸
    faces_dir = faces_root / stem
    faces_dir.mkdir(parents=True, exist_ok=True)
    print(f"[1] Tracking & cropping main face for {stem} → {faces_dir}")
    res = subprocess.run([
        sys.executable,
        str(script_dir / "track_main_subject.py"),
        "--frames_dir", str(frames_dir),
        "--out_dir",    str(faces_dir)
    ], check=False)
    if res.returncode != 0:
        print(f"[WARN] track_main_subject.py failed for {stem}, skipping EmoNet")
        return

    # 3) EmoNet 特征提取
    out_dir = features_root / stem
    print(f"[2] Predicting EmoNet for {stem} → {out_dir}")
    subprocess.run([
        sys.executable,
        str(script_dir / "batch_predict_emonet.py"),
        str(faces_dir),
        str(out_dir)
    ], check=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Track main face & run EmoNet (no frame extraction)")
    parser.add_argument("--videos_dir",     required=True,
                        help="Root folder containing your videos and the 'frames' subfolder")
    parser.add_argument("--frames_root",    default=None,
                        help="Where frames are stored (default: <videos_dir>/frames)")
    parser.add_argument("--faces_root",     default=None,
                        help="Where to save cropped faces (default: <videos_dir>/main_faces)")
    parser.add_argument("--features_root",  default=None,
                        help="Where to save EmoNet outputs (default: <videos_dir>/features)")

    args = parser.parse_args()
    vid_root      = Path(args.videos_dir).resolve()
    frames_root   = Path(args.frames_root  or vid_root / "frames").resolve()
    faces_root    = Path(args.faces_root   or vid_root / "main_faces").resolve()
    features_root = Path(args.features_root or vid_root / "features").resolve()
    script_dir    = Path(__file__).parent.resolve()

    # 视频文件夹下所有文件
    for ext in ("*.mp4", "*.mov", "*.avi", "*.mkv"):
        for video in sorted(vid_root.glob(ext)):
            process_video(video, frames_root, faces_root, features_root, script_dir)
