#检测哪些视频输出失败
import os

# 输入和输出路径
video_dir = "openpose/my_videos/naturalsocialtouch_datascience"
output_base = "openpose/output/processed"

# 获取所有视频名（不包含后缀）
video_files = sorted([
    f for f in os.listdir(video_dir)
    if f.endswith(".mov") or f.endswith(".mp4")
])

# 统计失败项
failed_videos = []

for video_file in video_files:
    video_name = os.path.splitext(video_file)[0]
    output_folder = os.path.join(output_base, video_name)
    output_json_folder = os.path.join(output_folder, "json")
    output_video_path = os.path.join(output_folder, "video.avi")

    # 条件 1：输出文件夹是否存在
    # 条件 2：json 文件夹内有至少一个 json 文件
    # 条件 3：video.avi 是否存在
    json_ok = os.path.exists(output_json_folder) and len(os.listdir(output_json_folder)) > 0
    video_ok = os.path.exists(output_video_path)

    if not (json_ok and video_ok):
        failed_videos.append(video_file)

# 输出结果
print(f"共检测到 {len(failed_videos)} 个处理失败的视频：")
for name in failed_videos:
    print(f"  - {name}")


#批量把文件名里的空格改成下划线，然后只重跑这些失败的
import os, shutil, subprocess, time

video_dir = "openpose/my_videos/naturalsocialtouch_datascience"
output_base = "openpose/output/processed"

failed = [
"017_RPReplay_Final1705855028 2.mov",
"070_RPReplay_Final1705977860 2.mov",
"075_RPReplay_Final1705977987 3.mov",
"110_RPReplay_Final1705988941 2.mov",
"121_RPReplay_Final1706026865 2.mov",
"169_RPReplay_Final1709913245 2.mov",
"170_RPReplay_Final1709913245 3.mov",
]

# 1) 重命名（空格->下划线）
renamed = []
for f in failed:
    src = os.path.join(video_dir, f)
    dst_name = f.replace(" ", "_")
    dst = os.path.join(video_dir, dst_name)
    if os.path.exists(src):
        os.rename(src, dst)
        renamed.append(dst_name)
        print("重命名:", f, "->", dst_name)
    else:
        print("未找到文件:", f)

# 2) 逐个重跑
for video_file in renamed:
    video_path = os.path.join(video_dir, video_file)
    video_name = os.path.splitext(video_file)[0]
    out_dir = os.path.join(output_base, video_name)
    out_json = os.path.join(out_dir, "json")
    out_avi  = os.path.join(out_dir, "video.avi")

    # 清理旧的失败产物
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_json, exist_ok=True)

    cmd = [
        "bash","-c",
        f"""
        cd openpose && ./build/examples/openpose/openpose.bin \
        --video "../{video_path}" \
        --write_json "../{out_json}" \
        --write_video "../{out_avi}" \
        --display 0 --render_pose 1
        """
    ]
    print("\n重处理:", video_file)
    t0=time.time()
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in p.stdout: print(line, end="")
    p.wait()
    print(f"完成 {video_name}，耗时 {time.time()-t0:.2f}s")
