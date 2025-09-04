import argparse
import json
import sys
from pathlib import Path

import cv2
import torch
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN
from torchvision import transforms

# 确保本地emonet包位于路径上
ROOT = Path(__file__).parent.resolve()
REPO = ROOT / "emonet"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from emonet.models.emonet import EmoNet

def predict_frame(detector, net, preprocess, device, img_path):
    # 读取图像
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        return None
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # 检测面部框
    boxes, _ = detector.detect(rgb)
    if boxes is None or len(boxes) == 0:
        return None
    x1, y1, x2, y2 = boxes[0].astype(int)

    h, w = rgb.shape[:2]
    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h))
    if x2 <= x1 or y2 <= y1:
        return None

    # 裁剪和预处理
    face = Image.fromarray(rgb[y1:y2, x1:x2])
    inp = preprocess(face).unsqueeze(0).to(device)

    with torch.no_grad():
        out = net(inp)

    return {
        "valence": float(out["valence"].clamp(-1, 1).cpu()),
        "arousal": float(out["arousal"].clamp(-1, 1).cpu())
    }

def main(input_dir, output_dir):
    INPUT = Path(input_dir)
    OUT   = Path(output_dir)

    if not INPUT.exists() or not INPUT.is_dir():
        print(f"ERROR: input_dir not found: {INPUT}")
        return

    faces = sorted(INPUT.glob("*.jpg"))
    if not faces:
        print(f"ERROR: no images in {INPUT}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化EmoNet
    net = EmoNet(n_expression=5)
    weights = REPO / "pretrained" / "emonet_5.pth"
    state = torch.load(weights, map_location=device)
    state = {k.replace("module.", ""): v for k, v in state.items()}
    net.load_state_dict(state, strict=False)
    net.to(device).eval()

    # 初始化检测器并预处理
    detector = MTCNN(keep_all=False, device=device)
    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    # 预测每张裁剪过的脸
    results = []
    for img_path in faces:
        r = predict_frame(detector, net, preprocess, device, img_path)
        if r is not None:
            results.append(r)

    if results:
        avg = {
            "frame_count": len(results),
            "valence": float(np.mean([r["valence"] for r in results])),
            "arousal": float(np.mean([r["arousal"] for r in results]))
        }
    else:
        avg = {"frame_count": 0, "valence": 0.0, "arousal": 0.0}

    out_json = OUT / "emonet.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(avg, f, indent=2)

    print(f"[Done] {INPUT.name} → {avg}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict EmoNet features on cropped face images"
    )
    parser.add_argument(
        "input_dir",
        help="Path to folder containing cropped face .jpg images"
    )
    parser.add_argument(
        "output_dir",
        help="Directory where emonet.json will be saved"
    )
    args = parser.parse_args()
    main(args.input_dir, args.output_dir)
