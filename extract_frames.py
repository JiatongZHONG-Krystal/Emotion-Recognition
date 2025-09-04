import cv2
import os
from pathlib import Path

def extract_all_frames(video_path: Path, output_folder: Path):

    output_folder.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[Error] 无法打开视频 {video_path}")
        return

    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_file = output_folder / f"{frame_id:05d}.jpg"
        cv2.imwrite(str(frame_file), frame)
        frame_id += 1

    cap.release()
    print(f"[Done] 从 {video_path.name} 提取了 {frame_id} 帧  {output_folder}")

def batch_extract(input_dir: str, output_root: str):
    """
    遍历 input_dir 下所有 .mp4/.avi/.mov 等视频文件，
    并为每个视频在 output_root 下创建同名子文件夹来保存帧。
    """
    video_dir = Path(input_dir)
    output_root = Path(output_root)

    # 支持的后缀，根据需要可以增删
    video_suffixes = {'.mp4', '.avi', '.mov', '.mkv'}

    for video_path in video_dir.iterdir():
        if video_path.suffix.lower() in video_suffixes:
            folder_name = video_path.stem  # 去掉扩展名的文件名
            out_folder = output_root / folder_name
            extract_all_frames(video_path, out_folder)

if __name__ == "__main__":
    # 你的原视频所在目录
    input_videos = r"D:\Data Science Research Project\naturalsocialtouch_datascience"
    # 想存放所有帧的根目录
    output_frames = r"D:\Data Science Research Project\naturalsocialtouch_datascience\frames"

    batch_extract(input_videos, output_frames)
