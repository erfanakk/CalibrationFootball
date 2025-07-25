import cv2
import torch
import logging
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path
from argparse import ArgumentParser
from eagle.models.keypoint_hrnet import KeypointModel
from eagle.utils.pitch import INTERSECTION_TO_PITCH_POINTS


def load_model(device: str):
    """Instantiate HRNet and load pretrained weights."""
    model = KeypointModel(57).to(device)
    w_path = Path(r"eagle/models/weights/weights/weights/keypoints_main.pth")
    model.load_state_dict(torch.load(w_path, map_location=device))
    model.eval()
    return model


def annotate_images(input_dir: str, output_dir: str, conf: float, show_label: bool, debug: bool, batch_size: int = 4):
    logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s",
                        level=logging.DEBUG if debug else logging.INFO)
    logger = logging.getLogger("KP-Image-Demo")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(device)
    logger.info(f"Using device: {device}")
    logger.info(f"Batch size: {batch_size}")

    transform = A.Compose([A.Resize(540, 960), A.Normalize(), ToTensorV2()])

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Supported image extensions
    exts = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
    image_files = [p for p in input_dir.iterdir() if p.suffix.lower() in exts]
    logger.info(f"Found {len(image_files)} images in {input_dir}")

    frame_idx, acc_time = 0, 0.0
    batch_imgs = []
    batch_tensors = []
    batch_paths = []

    for img_path in image_files:
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            logger.warning(f"Failed to read {img_path}")
            continue
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        tensor = transform(image=rgb)["image"]
        batch_imgs.append(img_bgr)
        batch_tensors.append(tensor)
        batch_paths.append(img_path.name)

        if len(batch_imgs) == batch_size or img_path == image_files[-1]:
            tic = torch.cuda.Event(enable_timing=True)
            toc = torch.cuda.Event(enable_timing=True)
            tic.record()

            batch_tensor = torch.stack(batch_tensors).to(device)
            batch_kp_lists = model.get_keypoints(batch_tensor)

            toc.record()
            torch.cuda.synchronize() if device == "cuda" else None
            elapsed = tic.elapsed_time(toc) / 1000.0 if device == "cuda" else 0.0
            if elapsed == 0.0:
                import time as _time
                elapsed = _time.perf_counter()

            for i, (img_bgr, kp_list, fname) in enumerate(zip(batch_imgs, batch_kp_lists, batch_paths)):
                height, width = img_bgr.shape[:2]
                for idx, x_n, y_n, score in kp_list:
                    if score < conf:
                        continue
                    x = int(x_n * width)
                    y = int(y_n * height)
                    cv2.circle(img_bgr, (x, y), 4, (0, 255, 255), -1)
                    if show_label:
                        label = INTERSECTION_TO_PITCH_POINTS[idx]
                        cv2.putText(img_bgr, label, (x + 5, y - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                                    (255, 255, 255), 1, cv2.LINE_AA)
                out_path = output_dir / fname
                cv2.imwrite(str(out_path), img_bgr)
                frame_idx += 1
            acc_time += elapsed
            batch_imgs.clear()
            batch_tensors.clear()
            batch_paths.clear()

    if frame_idx:
        logger.info(f"✔ Saved {frame_idx} images to {output_dir}")
        if acc_time > 0:
            logger.info(f"Avg processing FPS: {frame_idx / acc_time:5.2f}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input_dir", default=r"images", help="Input folder with images")
    parser.add_argument("--output_dir", default=r"out", help="Output folder for annotated images")
    parser.add_argument("--conf", type=float, default=0.10)
    parser.add_argument("--hide_label", action="store_true", default=False,
                        help="Do NOT print the point's text label")
    parser.add_argument("--debug", action="store_true", default=False,
                        help="Verbose timings & per-frame logs")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for processing (higher = better FPS, more memory)")
    args = parser.parse_args()

    annotate_images(args.input_dir,
                    args.output_dir,
                    conf=args.conf,
                    show_label=not args.hide_label,
                    debug=args.debug,
                    batch_size=args.batch_size) 

import cv2
import torch
import logging
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path
from argparse import ArgumentParser
from eagle.models.keypoint_hrnet import KeypointModel
from eagle.utils.pitch import INTERSECTION_TO_PITCH_POINTS, GROUND_TRUTH_POINTS

def load_model(device: str):
    """Instantiate HRNet and load pretrained weights."""
    model = KeypointModel(57).to(device)
    w_path = Path(r"eagle/models/weights/weights/weights/keypoints_main.pth")
    model.load_state_dict(torch.load(w_path, map_location=device))
    model.eval()
    return model

def annotate_images(input_dir: str, output_dir: str, conf: float, show_label: bool, debug: bool, batch_size: int = 4):
    """
    Process images to detect keypoints and draw pitch lines using homography.

    Args:
        input_dir (str): Directory containing input images.
        output_dir (str): Directory to save annotated images.
        conf (float): Confidence threshold for keypoint detection.
        show_label (bool): Whether to show keypoint labels on the image.
        debug (bool): Enable debug logging.
        batch_size (int): Number of images to process in a batch.
    """
    # Set up logging
    logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s",
                        level=logging.DEBUG if debug else logging.INFO)
    logger = logging.getLogger("Pitch-Line-Segmentation")

    # Initialize device and model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(device)
    logger.info(f"Using device: {device}")
    logger.info(f"Batch size: {batch_size}")

    # Define image transformation pipeline
    transform = A.Compose([A.Resize(540, 960), A.Normalize(), ToTensorV2()])

    # Set up input and output directories
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Supported image extensions
    exts = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
    image_files = [p for p in input_dir.iterdir() if p.suffix.lower() in exts]
    logger.info(f"Found {len(image_files)} images in {input_dir}")

    # Initialize batch processing variables
    frame_idx, acc_time = 0, 0.0
    batch_imgs = []
    batch_tensors = []
    batch_paths = []

    # Define canonical pitch lines (in meters, based on standard pitch dimensions)
    canonical_lines = [
        # Touchlines
        [(0, 0), (105, 0)],    # Bottom touchline
        [(0, 68), (105, 68)],  # Top touchline
        # Goal lines
        [(0, 0), (0, 68)],     # Left goal line
        [(105, 0), (105, 68)], # Right goal line
        # Halfway line
        [(52.5, 0), (52.5, 68)],
        # Left penalty area
        [(0, 13.84), (16.5, 13.84)],
        [(16.5, 13.84), (16.5, 54.16)],
        [(0, 54.16), (16.5, 54.16)],
        [(0, 13.84), (0, 54.16)],
        # Right penalty area
        [(105, 13.84), (88.5, 13.84)],
        [(88.5, 13.84), (88.5, 54.16)],
        [(105, 54.16), (88.5, 54.16)],
        [(105, 13.84), (105, 54.16)],
        # Left goal area
        [(0, 24.84), (5.5, 24.84)],
        [(5.5, 24.84), (5.5, 43.16)],
        [(0, 43.16), (5.5, 43.16)],
        [(0, 24.84), (0, 43.16)],
        # Right goal area
        [(105, 24.84), (99.5, 24.84)],
        [(99.5, 24.84), (99.5, 43.16)],
        [(105, 43.16), (99.5, 43.16)],
        [(105, 24.84), (105, 43.16)],
    ]

    # Center circle points (radius 9.15m, center at (52.5, 34.0))
    theta = np.linspace(0, 2*np.pi, 100)
    center = (52.5, 34.0)
    radius = 9.15
    circle_points = [(center[0] + radius * np.cos(t), center[1] + radius * np.sin(t)) for t in theta]

    for img_path in image_files:
        # Read and preprocess image
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            logger.warning(f"Failed to read {img_path}")
            continue
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        tensor = transform(image=rgb)["image"]
        batch_imgs.append(img_bgr)
        batch_tensors.append(tensor)
        batch_paths.append(img_path.name)

        # Process batch when full or on last image
        if len(batch_imgs) == batch_size or img_path == image_files[-1]:
            tic = torch.cuda.Event(enable_timing=True)
            toc = torch.cuda.Event(enable_timing=True)
            tic.record()

            # Run keypoint detection
            batch_tensor = torch.stack(batch_tensors).to(device)
            batch_kp_lists = model.get_keypoints(batch_tensor)

            toc.record()
            torch.cuda.synchronize() if device == "cuda" else None
            elapsed = tic.elapsed_time(toc) / 1000.0 if device == "cuda" else 0.0
            if elapsed == 0.0:
                import time as _time
                elapsed = _time.perf_counter()

            # Process each image in the batch
            for i, (img_bgr, kp_list, fname) in enumerate(zip(batch_imgs, batch_kp_lists, batch_paths)):
                height, width = img_bgr.shape[:2]

                # Draw detected keypoints
                for idx, x_n, y_n, score in kp_list:
                    if score < conf:
                        continue
                    x = int(x_n * width)
                    y = int(y_n * height)
                    cv2.circle(img_bgr, (x, y), 4, (0, 255, 255), -1)  # Yellow circles for keypoints
                    if show_label:
                        label = INTERSECTION_TO_PITCH_POINTS[idx]
                        cv2.putText(img_bgr, label, (x + 5, y - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                                    (255, 255, 255), 1, cv2.LINE_AA)

                # Filter keypoints for homography estimation (exclude non-planar points)
                valid_kp = [kp for kp in kp_list if kp[3] > conf and kp[0] not in [0, 1, 24, 25]]
                if len(valid_kp) < 4:
                    logger.warning(f"Not enough keypoints for homography in {fname}")
                else:
                    # Prepare source (canonical) and destination (image) points
                    src_pts = []
                    dst_pts = []
                    for idx, x_n, y_n, score in valid_kp:
                        label = INTERSECTION_TO_PITCH_POINTS[idx]
                        X, Y, Z = GROUND_TRUTH_POINTS[label]
                        if Z != 0:
                            continue  # Skip non-planar points
                        src_pts.append([X, Y])
                        x = x_n * width
                        y = y_n * height
                        dst_pts.append([x, y])

                    if len(src_pts) >= 4:
                        src_pts = np.array(src_pts, dtype=np.float32)
                        dst_pts = np.array(dst_pts, dtype=np.float32)

                        # Estimate homography
                        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                        if H is not None:
                            # Project and draw straight lines
                            for line in canonical_lines:
                                p1 = np.array(line[0], dtype=np.float32).reshape(-1, 1, 2)
                                p2 = np.array(line[1], dtype=np.float32).reshape(-1, 1, 2)
                                proj_p1 = cv2.perspectiveTransform(p1, H)
                                proj_p2 = cv2.perspectiveTransform(p2, H)
                                pt1 = tuple(proj_p1[0][0].astype(int))
                                pt2 = tuple(proj_p2[0][0].astype(int))
                                cv2.line(img_bgr, pt1, pt2, (0, 255, 0), 2)  # Green lines

                            # Project and draw center circle
                            circle_points_array = np.array(circle_points, dtype=np.float32).reshape(-1, 1, 2)
                            proj_circle_points = cv2.perspectiveTransform(circle_points_array, H)
                            pts = proj_circle_points.reshape(-1, 2).astype(int)
                            cv2.polylines(img_bgr, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
                        else:
                            logger.warning(f"Homography estimation failed for {fname}")
                    else:
                        logger.warning(f"Not enough planar points for homography in {fname}")

                # Save the annotated image
                out_path = output_dir / fname
                cv2.imwrite(str(out_path), img_bgr)
                frame_idx += 1

            # Update timing and clear batch
            acc_time += elapsed
            batch_imgs.clear()
            batch_tensors.clear()
            batch_paths.clear()

    # Log summary
#    if frame_idx:
#         logger.info(f"✔ Saved {frame_idx} images to {output_dir}")
#         if acc_time > 0:
#             logger.info(f"Avg processing FPS: {frame_idx / acc_time:5.2f}")

if __name__ == "__main__":
    parser = ArgumentParser(description="Detect keypoints and draw soccer pitch lines on images.")
    parser.add_argument("--input_dir", default="images", help="Input folder with images")
    parser.add_argument("--output_dir", default="out3", help="Output folder for annotated images")
    parser.add_argument("--conf", type=float, default=0.1, help="Confidence threshold for keypoints")
    parser.add_argument("--hide_label", action="store_true", default=False,
                        help="Do NOT print the point's text label")
    parser.add_argument("--debug", action="store_true", default=False,
                        help="Verbose timings & per-frame logs")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for processing (higher = better FPS, more memory)")
    args = parser.parse_args()

    annotate_images(args.input_dir,
                    args.output_dir,
                    conf=args.conf,
                    show_label=not args.hide_label,
                    debug=args.debug,
                    batch_size=args.batch_size)