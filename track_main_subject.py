#!/usr/bin/env python3
import argparse, sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from facenet_pytorch import MTCNN
from deep_sort_realtime.deepsort_tracker import DeepSort

def main(frames_dir, out_dir, top_k):
    frames_dir = Path(frames_dir)
    out_dir    = Path(out_dir)
    if not frames_dir.exists():
        sys.exit(f"ERROR: no frames_dir: {frames_dir}")
    imgs = sorted(frames_dir.glob("*.jpg"), key=lambda p: p.name)
    if not imgs:
        sys.exit(f"ERROR: no images in {frames_dir}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    mtcnn  = MTCNN(keep_all=True, device=device)
    tracker= DeepSort(max_age=30)

    # 用于跟踪、记录每帧的轨迹框
    frame_tracks  = []
    track_frames  = {}
    motion_energy = {}

    prev_gray = None

    # 逐帧处理
    for idx, img_path in enumerate(imgs):
        bgr = cv2.imread(str(img_path))
        gray= cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        #计算像素差分
        if prev_gray is not None:
            diff = cv2.absdiff(gray, prev_gray)
        else:
            diff = None
        prev_gray = gray

        #MTCNN 检人脸
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        boxes, probs = mtcnn.detect(rgb)
        dets = []
        if boxes is not None:
            for box, prob in zip(boxes, probs):
                x1,y1,x2,y2 = box.astype(int)
                dets.append(([x1,y1,x2-x1,y2-y1], float(prob), "face"))

        #DeepSort 跟踪
        tracks = tracker.update_tracks(dets, frame=bgr)
        this_frame = []
        for tr in tracks:
            if not tr.is_confirmed():
                continue
            tid = tr.track_id
            l,t,r,b = map(int, tr.to_ltrb())
            this_frame.append((tid,l,t,r,b))
            track_frames.setdefault(tid, []).append(idx)
            motion_energy.setdefault(tid, 0.0)
            # 在框内累加差分能量
            if diff is not None and r>l and b>t:
                roi = diff[t:b, l:r]
                motion_energy[tid] += float(np.sum(roi))
        frame_tracks.append(this_frame)

    if not motion_energy:
        sys.exit("ERROR: no face tracks found!")

    #按运动能量选前 K 条轨迹
    top = sorted(motion_energy.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    selected = [tid for tid,_ in top]
    print(f"Selected active track IDs: {selected}")

    #裁剪这 K 条轨迹的所有人脸
    for tid in selected:
        subdir = out_dir/f"track_{tid}"
        subdir.mkdir(parents=True, exist_ok=True)
        saved = 0
        for idx, img_path in enumerate(imgs):
            for id2,l,t,r,b in frame_tracks[idx]:
                if id2!=tid:
                    continue
                crop = Image.open(img_path).crop((l,t,r,b))
                crop.save(subdir/f"{img_path.stem}.jpg")
                saved +=1
                break
        print(f"→ track {tid}: saved {saved} crops → {subdir}")

if __name__=="__main__":
    p = argparse.ArgumentParser(
        description="Select top-K active faces by pixel-diff motion energy"
    )
    p.add_argument("--frames_dir", required=True,
                   help="Folder of extracted frames (*.jpg)")
    p.add_argument("--out_dir",    required=True,
                   help="Where to save cropped faces (subdirs track_ID)")
    p.add_argument("--top_k",      type=int, default=2,
                   help="Number of active tracks to keep")
    args = p.parse_args()
    main(args.frames_dir, args.out_dir, args.top_k)
