# 批量读取所有视频
import os
import subprocess
import time
import shutil

video_dir = "openpose/my_videos/naturalsocialtouch_datascience"
output_base = "openpose/output/processed"

# 获取所有视频文件
video_files = sorted([
    f for f in os.listdir(video_dir)
    if f.endswith(".mp4") or f.endswith(".mov")
])

print(f"共检测到 {len(video_files)} 个视频待处理。\n")

for idx, video_file in enumerate(video_files, 1):
    video_path = os.path.join(video_dir, video_file)
    video_name = os.path.splitext(video_file)[0]

    output_folder = os.path.join(output_base, video_name)
    output_json_folder = os.path.join(output_folder, "json")
    output_video_path = os.path.join(output_folder, "video.avi")

    # 如果输出视频已存在，则跳过
    if os.path.exists(output_video_path):
        print(f"[{idx}/{len(video_files)}] 跳过 {video_file}（已处理）")
        continue

    # 清理旧的空文件夹
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)

    os.makedirs(output_json_folder, exist_ok=True)

    # 构造命令
    command = [
        "bash", "-c",
        f"""
        cd openpose && ./build/examples/openpose/openpose.bin \
        --video ../{video_path} \
        --write_json ../{output_json_folder} \
        --write_video ../{output_video_path} \
        --display 0 --render_pose 1
        """
    ]

    print(f"\n[{idx}/{len(video_files)}] 处理: {video_file}")
    print("=" * 60)
    start_time = time.time()

    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in process.stdout:
        print(line, end='')
    process.wait()

    elapsed = time.time() - start_time
    print(f"\n 完成 {video_name}，耗时 {elapsed:.2f} 秒")
    print("=" * 60)


#遍历所有视频输出json和渲染后视频
import os
import subprocess
import time

video_dir = "openpose/my_videos/naturalsocialtouch_datascience"
output_base = "openpose/output/processed"

video_files = sorted([
    f for f in os.listdir(video_dir) if f.endswith(".mov") or f.endswith(".mp4")
])

for idx, video_file in enumerate(video_files, 1):
    video_path = os.path.join(video_dir, video_file)
    video_name = os.path.splitext(video_file)[0]

    output_folder = os.path.join(output_base, video_name)
    output_json_folder = os.path.join(output_folder, "json")
    output_video_path = os.path.join(output_folder, "video.avi")

    if os.path.exists(output_video_path) and \
       os.path.isdir(output_json_folder) and len(os.listdir(output_json_folder)) > 0:
        print(f" 跳过已完成：{video_file}\n{'-'*60}")
        continue

    os.makedirs(output_json_folder, exist_ok=True)

    command = [
        "bash", "-c",
        f"""
        cd openpose && ./build/examples/openpose/openpose.bin \
        --video ../{video_path} \
        --write_json ../{output_json_folder} \
        --write_video ../{output_video_path} \
        --display 0 --render_pose 1
        """
    ]

    print(f"\n[{idx}/{len(video_files)}] 处理：{video_file}")
    print("=" * 60)
    start_time = time.time()

    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in process.stdout:
        print(line, end='')
    process.wait()

    elapsed = time.time() - start_time
    print(f"\n 完成 {video_name}，耗时 {elapsed:.2f} 秒")
    print("=" * 60)