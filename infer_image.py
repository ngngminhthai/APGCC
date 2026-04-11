"""
APGCC inference script — mirrors infer_image_old.py (P2PNet) for the APGCC model.

Usage:
    python infer_image.py \
        --config   apgcc/configs/steelbar_train.yml \
        --checkpoint output/steelbar/best.pth \
        --image    path/to/image.jpg \
        --output   inference_result.jpg \
        --json     inference_result.json \
        [--threshold 0.5] \
        [--slice_size 512] \
        [--slice_overlap 128] \
        [--dedup_dist 20]
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

# ── make sure apgcc/ is importable ───────────────────────────────────────────
APGCC_DIR = Path(__file__).parent / "apgcc"
if str(APGCC_DIR) not in sys.path:
    sys.path.insert(0, str(APGCC_DIR))

from models import build_model
from models.APGCC import NestedTensor
from config import cfg, merge_from_file


# ── constants ─────────────────────────────────────────────────────────────────
POINT_RADIUS = 4
POINT_COLOR_BGR = (0, 0, 255)
TEXT_COLOR_BGR = (255, 255, 255)
TEXT_BG_COLOR_BGR = (0, 0, 0)


# ── helpers ───────────────────────────────────────────────────────────────────

def load_config(config_path: str):
    return merge_from_file(cfg, config_path)


def load_model(cfg, checkpoint_path: str, device: torch.device):
    model = build_model(cfg=cfg, training=False)
    model.to(device)

    state = torch.load(checkpoint_path, map_location=device)
    # support plain state-dict or {'model': state_dict} wrappers
    if isinstance(state, dict) and "model" in state:
        state = state["model"]

    # strip DataParallel / DDP prefix
    cleaned = {k.replace("module.", "", 1): v for k, v in state.items()}
    model.load_state_dict(cleaned, strict=True)
    model.eval()
    return model


def preprocess_pil(image: Image.Image) -> NestedTensor:
    """Convert a PIL crop to a NestedTensor (batch=1, mask=all-False)."""
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    tensor = transform(image).unsqueeze(0)          # (1, 3, H, W)
    mask = torch.zeros(
        (1, tensor.shape[2], tensor.shape[3]),
        dtype=torch.bool,
    )                                               # (1, H, W)  — no padding
    return NestedTensor(tensor, mask)


def generate_slices(image_w: int, image_h: int, slice_size: int, overlap: int):
    """Yield (x0, y0, x1, y1) tiles covering the full image."""
    stride = slice_size - overlap
    xs = list(range(0, image_w, stride))
    ys = list(range(0, image_h, stride))
    for y0 in ys:
        for x0 in xs:
            x1 = min(x0 + slice_size, image_w)
            y1 = min(y0 + slice_size, image_h)
            x0 = max(0, x1 - slice_size)
            y0 = max(0, y1 - slice_size)
            yield x0, y0, x1, y1


def deduplicate_points(
    points: np.ndarray,
    scores: np.ndarray,
    min_dist: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Suppress duplicate detections from overlapping slices."""
    if len(points) == 0:
        return points, scores

    order = np.argsort(-scores)
    points = points[order]
    scores = scores[order]

    keep = np.ones(len(points), dtype=bool)
    for i in range(len(points)):
        if not keep[i]:
            continue
        dx = points[i, 0] - points[i + 1:, 0]
        dy = points[i, 1] - points[i + 1:, 1]
        too_close = dx * dx + dy * dy < min_dist * min_dist
        keep[i + 1:][too_close] = False

    return points[keep], scores[keep]


@torch.no_grad()
def predict_slice(
    model,
    nested: NestedTensor,
    device: torch.device,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    nested = nested.to(device)
    outputs = model(nested)

    scores = torch.softmax(outputs["pred_logits"], dim=-1)[0, :, 1]
    points = outputs["pred_points"][0]

    keep = scores > threshold
    kept_scores = scores[keep].cpu().numpy()
    kept_points = points[keep].cpu().numpy()
    return kept_points, kept_scores


def predict_sliding_window(
    model,
    image: Image.Image,
    device: torch.device,
    threshold: float,
    slice_size: int,
    overlap: int,
    dedup_dist: int,
) -> tuple[np.ndarray, np.ndarray]:
    image_w, image_h = image.size
    slices = list(generate_slices(image_w, image_h, slice_size, overlap))
    print(
        f"Image: {image_w}x{image_h} — {len(slices)} slices "
        f"({slice_size}x{slice_size}, overlap={overlap})"
    )

    all_points, all_scores = [], []
    for idx, (x0, y0, x1, y1) in enumerate(slices):
        crop = image.crop((x0, y0, x1, y1))
        nested = preprocess_pil(crop)

        pts, scs = predict_slice(model, nested, device, threshold)
        if len(pts) > 0:
            pts[:, 0] += x0
            pts[:, 1] += y0
            all_points.append(pts)
            all_scores.append(scs)

        if (idx + 1) % 20 == 0 or (idx + 1) == len(slices):
            print(f"  Processed {idx + 1}/{len(slices)} slices …")

    if not all_points:
        return np.empty((0, 2), np.float32), np.empty((0,), np.float32)

    all_points = np.concatenate(all_points, axis=0)
    all_scores = np.concatenate(all_scores, axis=0)
    all_points, all_scores = deduplicate_points(all_points, all_scores, dedup_dist)
    return all_points, all_scores


def draw_points(image: Image.Image, points: np.ndarray, scores: np.ndarray) -> np.ndarray:
    image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    for point, score in zip(points, scores):
        x, y = int(round(point[0])), int(round(point[1]))
        cv2.circle(image_bgr, (x, y), POINT_RADIUS, POINT_COLOR_BGR, -1)
        cv2.putText(
            image_bgr,
            f"{score:.2f}",
            (x + 6, y - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            TEXT_COLOR_BGR,
            1,
            cv2.LINE_AA,
        )

    count_text = f"Count: {len(points)}"
    (tw, th), _ = cv2.getTextSize(count_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    cv2.rectangle(image_bgr, (10, 10), (20 + tw, 20 + th), TEXT_BG_COLOR_BGR, -1)
    cv2.putText(
        image_bgr,
        count_text,
        (15, 15 + th),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        TEXT_COLOR_BGR,
        2,
        cv2.LINE_AA,
    )
    return image_bgr


def save_json(points: np.ndarray, scores: np.ndarray, path: Path, threshold: float,
              slice_size: int, overlap: int):
    payload = {
        "count": int(len(points)),
        "threshold": threshold,
        "slice_size": slice_size,
        "slice_overlap": overlap,
        "points": [
            {"x": float(p[0]), "y": float(p[1]), "score": float(s)}
            for p, s in zip(points, scores)
        ],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="APGCC single-image inference")
    parser.add_argument("-c", "--config",     required=True,  help="Path to .yml config (e.g. apgcc/configs/steelbar_train.yml)")
    parser.add_argument("-w", "--checkpoint", required=True,  help="Path to .pth checkpoint")
    parser.add_argument("-i", "--image",      required=True,  help="Path to input image")
    parser.add_argument("-o", "--output",     default="inference_result.jpg", help="Output image path")
    parser.add_argument("-j", "--json",       default="inference_result.json", help="Output JSON path")
    parser.add_argument("--threshold",    type=float, default=0.5,   help="Score threshold (default: 0.5)")
    parser.add_argument("--slice_size",   type=int,   default=512,   help="Sliding-window tile size (default: 512)")
    parser.add_argument("--slice_overlap",type=int,   default=128,   help="Tile overlap in pixels (default: 128)")
    parser.add_argument("--dedup_dist",   type=int,   default=20,    help="Dedup distance in pixels (default: 20)")
    return parser.parse_args()


def main():
    args = parse_args()

    image_path      = Path(args.image)
    checkpoint_path = Path(args.checkpoint)
    output_image    = Path(args.output)
    output_json     = Path(args.json)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    output_image.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    cfg_loaded = load_config(args.config)
    model = load_model(cfg_loaded, str(checkpoint_path), device)

    image = Image.open(image_path).convert("RGB")
    points, scores = predict_sliding_window(
        model, image, device,
        threshold=args.threshold,
        slice_size=args.slice_size,
        overlap=args.slice_overlap,
        dedup_dist=args.dedup_dist,
    )

    drawn = draw_points(image, points, scores)
    cv2.imwrite(str(output_image), drawn)
    save_json(points, scores, output_json, args.threshold, args.slice_size, args.slice_overlap)

    print(f"Input image  : {image_path}")
    print(f"Checkpoint   : {checkpoint_path}")
    print(f"Predicted cnt: {len(points)}")
    print(f"Saved image  : {output_image}")
    print(f"Saved JSON   : {output_json}")


if __name__ == "__main__":
    main()
