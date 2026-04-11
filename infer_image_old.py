import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from models import build_model


# Hardcoded config
CHECKPOINT_PATH = Path(r"C:\Users\User\Downloads\CrowdCounting-P2PNet\best_mae.pth")
IMAGE_PATH = Path(r"C:\Users\User\Downloads\00000815429100048316.Image.235000-test.png")
OUTPUT_IMAGE_PATH = Path(r"C:\Users\User\Downloads\CrowdCounting-P2PNet\inference_result.jpg")
OUTPUT_JSON_PATH = Path(r"C:\Users\User\Downloads\CrowdCounting-P2PNet\inference_result.json")

BACKBONE = "vgg16_bn"
ROW = 2
LINE = 2
THRESHOLD = 0.3
POINT_RADIUS = 4
POINT_COLOR_BGR = (0, 0, 255)
TEXT_COLOR_BGR = (255, 255, 255)
TEXT_BG_COLOR_BGR = (0, 0, 0)

SLICE_SIZE = 512
SLICE_OVERLAP = 128  # overlap between slices to avoid missing detections at edges
DEDUP_MIN_DIST = 20  # minimum pixel distance between two kept points; closer ones are merged


def build_config():
    return SimpleNamespace(
        backbone=BACKBONE,
        row=ROW,
        line=LINE,
    )


def load_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint

    # Handle checkpoints saved from DataParallel/DDP.
    cleaned_state_dict = {}
    for key, value in state_dict.items():
        cleaned_state_dict[key.replace("module.", "", 1)] = value

    model.load_state_dict(cleaned_state_dict, strict=True)
    return checkpoint


def preprocess_pil(image: Image.Image):
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0)


def generate_slices(image_w, image_h, slice_size, overlap):
    """Yield (x0, y0, x1, y1) for each slice tile covering the full image."""
    stride = slice_size - overlap
    xs = list(range(0, image_w, stride))
    ys = list(range(0, image_h, stride))
    for y0 in ys:
        for x0 in xs:
            x1 = min(x0 + slice_size, image_w)
            y1 = min(y0 + slice_size, image_h)
            # Shift window left/up if it's smaller than slice_size (edge tiles)
            x0 = max(0, x1 - slice_size)
            y0 = max(0, y1 - slice_size)
            yield x0, y0, x1, y1


def deduplicate_points(points, scores, min_dist=DEDUP_MIN_DIST):
    """Remove duplicate detections that are very close together (from overlapping slices)."""
    if len(points) == 0:
        return points, scores

    keep = np.ones(len(points), dtype=bool)
    # Sort by score descending so higher-confidence detections are kept
    order = np.argsort(-scores)
    points = points[order]
    scores = scores[order]

    for i in range(len(points)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(points)):
            if not keep[j]:
                continue
            dx = points[i][0] - points[j][0]
            dy = points[i][1] - points[j][1]
            if dx * dx + dy * dy < min_dist * min_dist:
                keep[j] = False

    return points[keep], scores[keep]


@torch.no_grad()
def predict_points_on_tensor(model, image_tensor, device):
    image_tensor = image_tensor.to(device)
    outputs = model(image_tensor)
    scores = torch.softmax(outputs["pred_logits"], dim=-1)[0, :, 1]
    points = outputs["pred_points"][0]

    keep = scores > THRESHOLD
    kept_scores = scores[keep].detach().cpu().numpy()
    kept_points = points[keep].detach().cpu().numpy()

    return kept_points, kept_scores


def predict_points_sliding_window(model, image: Image.Image, device):
    """Run inference on 512x512 slices and remap points to original image coordinates."""
    image_w, image_h = image.size
    all_points = []
    all_scores = []

    slices = list(generate_slices(image_w, image_h, SLICE_SIZE, SLICE_OVERLAP))
    print(f"Image size: {image_w}x{image_h} — {len(slices)} slices ({SLICE_SIZE}x{SLICE_SIZE}, overlap={SLICE_OVERLAP})")

    for idx, (x0, y0, x1, y1) in enumerate(slices):
        slice_img = image.crop((x0, y0, x1, y1))
        tensor = preprocess_pil(slice_img)

        points, scores = predict_points_on_tensor(model, tensor, device)

        # Remap local slice coordinates → original image coordinates
        if len(points) > 0:
            points[:, 0] += x0
            points[:, 1] += y0
            all_points.append(points)
            all_scores.append(scores)

        if (idx + 1) % 20 == 0 or (idx + 1) == len(slices):
            print(f"  Processed {idx + 1}/{len(slices)} slices …")

    if len(all_points) == 0:
        return np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=np.float32)

    all_points = np.concatenate(all_points, axis=0)
    all_scores = np.concatenate(all_scores, axis=0)

    all_points, all_scores = deduplicate_points(all_points, all_scores)
    return all_points, all_scores


def draw_points(image, points, scores):
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
    text_size, _ = cv2.getTextSize(count_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    cv2.rectangle(image_bgr, (10, 10), (20 + text_size[0], 20 + text_size[1]), TEXT_BG_COLOR_BGR, -1)
    cv2.putText(
        image_bgr,
        count_text,
        (15, 15 + text_size[1]),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        TEXT_COLOR_BGR,
        2,
        cv2.LINE_AA,
    )

    return image_bgr


def save_points(points, scores, output_json_path):
    payload = {
        "count": int(len(points)),
        "threshold": THRESHOLD,
        "slice_size": SLICE_SIZE,
        "slice_overlap": SLICE_OVERLAP,
        "points": [
            {"x": float(point[0]), "y": float(point[1]), "score": float(score)}
            for point, score in zip(points, scores)
        ],
    }
    output_json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main():
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")
    if not IMAGE_PATH.exists():
        raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

    OUTPUT_IMAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(build_config(), training=False)
    model.to(device)
    load_checkpoint(model, CHECKPOINT_PATH, device)
    model.eval()

    original_image = Image.open(IMAGE_PATH).convert("RGB")
    points, scores = predict_points_sliding_window(model, original_image, device)

    drawn = draw_points(original_image, points, scores)
    cv2.imwrite(str(OUTPUT_IMAGE_PATH), drawn)
    save_points(points, scores, OUTPUT_JSON_PATH)

    print(f"Input image: {IMAGE_PATH}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Predicted count: {len(points)}")
    print(f"Saved image: {OUTPUT_IMAGE_PATH}")
    print(f"Saved points: {OUTPUT_JSON_PATH}")


if __name__ == "__main__":
    main()