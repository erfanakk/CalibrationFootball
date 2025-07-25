# detect_and_segment.py  -------------------------------------------------------
"""
End‑to‑end demo:
  1. HRNet key‑point detection (57 intersections)
  2. Homography estimation + pitch/goal segmentation
  3. Overlay visualisation and mask export

Usage:
    python detect_and_segment.py --frame path/to/frame.jpg --out overlay.png
"""

from __future__ import annotations
import cv2
import torch
import numpy as np
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
import argparse
import logging
from typing import List, Tuple, Dict

# -----------------------------------------------------------------------------#
# -------------------------  1.  Key‑point detection  -------------------------#
# -----------------------------------------------------------------------------#
from eagle.models.keypoint_hrnet import KeypointModel           # your model
from eagle.utils.pitch import INTERSECTION_TO_PITCH_POINTS      # idx → name

LOGGER = logging.getLogger("Detect&Segment")
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO
)


def load_model(device: str) -> torch.nn.Module:
    """Instantiate HRNet and load pretrained weights (same as in your script)."""
    model = KeypointModel(57).to(device)
    w_path = (
        Path(__file__).parent
        / "eagle/models/weights/weights/weights/keypoints_main.pth"
    )
    model.load_state_dict(torch.load(w_path, map_location=device))
    model.eval()
    return model


# Albumentations pipeline (matches training preprocessing)
TRANSFORM = A.Compose(
    [
        A.Resize(540, 960),
        A.Normalize(),
        ToTensorV2(),
    ]
)

# -----------------------------------------------------------------------------#
# ------------------------- 2.  Template‑Matching code ------------------------#
# -----------------------------------------------------------------------------#
import math
from itertools import pairwise       # py ≥ 3.10

# --- template coordinates ----------------------------------------------------
from eagle.utils.pitch import (
    LR_SIDES_MAPPING, TOP_BOTTOM_MAPPING, NOT_ON_PLANE,
    GROUND_TRUTH_POINTS_NORMALIZED, INTERSECTION_TO_PITCH_POINTS,
)

PITCH_POINTS_TO_INTERSECTION = {v: k for k, v in INTERSECTION_TO_PITCH_POINTS.items()}

TEMPLATE_PT = {
    name: (x, y) for name, (x, y, _) in GROUND_TRUTH_POINTS_NORMALIZED.items()
}

# ------------------------------------------------------------------ helpers --
def _metric_to_norm(x_m: float, y_m: float) -> tuple[float, float]:
    """Convert metre coordinates (UEFA pitch 105×68 m) → 0‑100 normalised."""
    return x_m / 105.0 * 100.0, y_m / 68.0 * 100.0


def _ellipse_chords(
    centre_m: tuple[float, float],
    radius_m: float,
    angle_start_deg: float,
    angle_end_deg: float,
    n: int,
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """
    Return `n` short line‑segments approximating a circle/arc in template coords.
    Angles are given in **degrees**, counter‑clockwise, 0  ° = +X axis.
    """
    cx_m, cy_m = centre_m
    chords = []
    for a0, a1 in pairwise(
        np.linspace(math.radians(angle_start_deg),
                    math.radians(angle_end_deg),
                    n + 1)
    ):
        for ang0, ang1 in [(a0, a1)]:
            x0_m = cx_m + radius_m * math.cos(ang0)
            y0_m = cy_m + radius_m * math.sin(ang0)
            x1_m = cx_m + radius_m * math.cos(ang1)
            y1_m = cy_m + radius_m * math.sin(ang1)
            chords.append((_metric_to_norm(x0_m, y0_m),
                           _metric_to_norm(x1_m, y1_m)))
    return chords


# ------------------------------------------------------------------ segments --
LINE_NAME_SEGMENTS: list[tuple[str, str]] = [
    # outer rectangle
    ("TL_PITCH_CORNER", "TR_PITCH_CORNER"),
    ("TR_PITCH_CORNER", "BR_PITCH_CORNER"),
    ("BR_PITCH_CORNER", "BL_PITCH_CORNER"),
    ("BL_PITCH_CORNER", "TL_PITCH_CORNER"),
    # halfway
    ("T_TOUCH_AND_HALFWAY_LINES_INTERSECTION",
     "B_TOUCH_AND_HALFWAY_LINES_INTERSECTION"),
    # left + right penalty areas
    ("L_PENALTY_AREA_TL_CORNER", "L_PENALTY_AREA_TR_CORNER"),
    ("L_PENALTY_AREA_TR_CORNER", "L_PENALTY_AREA_BR_CORNER"),
    ("L_PENALTY_AREA_BR_CORNER", "L_PENALTY_AREA_BL_CORNER"),
    ("L_PENALTY_AREA_BL_CORNER", "L_PENALTY_AREA_TL_CORNER"),
    ("R_PENALTY_AREA_TL_CORNER", "R_PENALTY_AREA_TR_CORNER"),
    ("R_PENALTY_AREA_TR_CORNER", "R_PENALTY_AREA_BR_CORNER"),
    ("R_PENALTY_AREA_BR_CORNER", "R_PENALTY_AREA_BL_CORNER"),
    ("R_PENALTY_AREA_BL_CORNER", "R_PENALTY_AREA_TL_CORNER"),
    # goal areas
    ("L_GOAL_AREA_TL_CORNER", "L_GOAL_AREA_TR_CORNER"),
    ("L_GOAL_AREA_TR_CORNER", "L_GOAL_AREA_BR_CORNER"),
    ("L_GOAL_AREA_BR_CORNER", "L_GOAL_AREA_BL_CORNER"),
    ("L_GOAL_AREA_BL_CORNER", "L_GOAL_AREA_TL_CORNER"),
    ("R_GOAL_AREA_TL_CORNER", "R_GOAL_AREA_TR_CORNER"),
    ("R_GOAL_AREA_TR_CORNER", "R_GOAL_AREA_BR_CORNER"),
    ("R_GOAL_AREA_BR_CORNER", "R_GOAL_AREA_BL_CORNER"),
    ("R_GOAL_AREA_BL_CORNER", "R_GOAL_AREA_TL_CORNER"),
    # goal mouths on grass
    ("L_GOAL_BL_POST", "L_GOAL_BR_POST"),
    ("L_GOAL_BR_POST", "L_GOAL_TR_POST"),
    ("L_GOAL_TR_POST", "L_GOAL_TL_POST"),
    ("L_GOAL_TL_POST", "L_GOAL_BL_POST"),
    ("R_GOAL_BL_POST", "R_GOAL_BR_POST"),
    ("R_GOAL_BR_POST", "R_GOAL_TR_POST"),
    ("R_GOAL_TR_POST", "R_GOAL_TL_POST"),
    ("R_GOAL_TL_POST", "R_GOAL_BL_POST"),
]

# --- curved markings ---------------------------------------------------------
CHORDS_CURVED: list[tuple[tuple[float, float], tuple[float, float]]] = []

# centre circle (full 360°)
CHORDS_CURVED += _ellipse_chords(
    (52.5, 34.0),           # centre‑spot in metres
    9.15,                   # radius in metres
    0, 360,
    n=60,                   # finer → smoother
)

# left penalty arc (outside penalty box → faces centre line)
CHORDS_CURVED += _ellipse_chords(
    (11.0, 34.0),           # L_PENALTY_MARK
    9.15,
    -60, 60,                # ±60° span
    n=20,
)
# right penalty arc
CHORDS_CURVED += _ellipse_chords(
    (94.0, 34.0),           # R_PENALTY_MARK
    9.15,
    120, 240,               # 180±60°
    n=20,
)

# --- spot marks (centre & penalties) -----------------------------------------
SPOT_MARKS_METRIC = [
    (52.5, 34.0),   # centre mark
    (11.0, 34.0),   # left penalty mark
    (94.0, 34.0),   # right penalty mark
]
SPOT_MARKS_NORM = [_metric_to_norm(x, y) for x, y in SPOT_MARKS_METRIC]


# ==========================  TemplateMatcher class  ===========================
class TemplateMatcher:
    """
    Homography estimation + rasterise every white marking (lines, arcs, spots).
    """

    def __init__(self,
                 ransac_thresh: float = 5.0,
                 min_inliers: int = 10) -> None:
        self.ransac_thresh = ransac_thresh
        self.min_inliers = min_inliers
        self.H: np.ndarray | None = None
        self.orientation: str | None = None  # "none" | "lr" | "tb" | "lr_tb"

    # ----------------------------- helpers ----------------------------------
    @staticmethod
    def _remap(name: str, orientation: str) -> str:
        if orientation in ("lr", "lr_tb"):
            name = LR_SIDES_MAPPING.get(name, name)
        if orientation in ("tb", "lr_tb"):
            name = TOP_BOTTOM_MAPPING.get(name, name)
        return name

    def _build_xy(self, detected: dict[int, tuple[float, float]], ori: str):
        src, dst = [], []
        for idx, (u, v) in detected.items():
            if idx in NOT_ON_PLANE:
                continue
            name = self._remap(INTERSECTION_TO_PITCH_POINTS[idx], ori)
            if name not in TEMPLATE_PT:
                continue
            src.append(TEMPLATE_PT[name])
            dst.append((u, v))
        return np.asarray(src, np.float32), np.asarray(dst, np.float32)

    # ----------------------------- API --------------------------------------
    def estimate(self, detected: dict[int, tuple[float, float]]) -> bool:
        best_H, best_inl, best_ori = None, -1, None
        for ori in ("none", "lr", "tb", "lr_tb"):
            s, d = self._build_xy(detected, ori)
            if len(s) < 4:
                continue
            H, m = cv2.findHomography(s, d, cv2.RANSAC, self.ransac_thresh)
            if H is None:
                continue
            inl = int(m.sum())
            if inl > best_inl:
                best_H, best_inl, best_ori = H, inl, ori
        if best_H is not None and best_inl >= self.min_inliers:
            self.H, self.orientation = best_H, best_ori
            logging.getLogger("Detect&Segment").info(
                f"H found: ori={best_ori}, inliers={best_inl}"
            )
            return True
        return False

    # ------------------------------------------------------------------
    def _proj(self, x: float, y: float) -> tuple[int, int]:
        p = cv2.perspectiveTransform(np.array([[[x, y]]], np.float32), self.H)[0][0]
        return int(round(p[0])), int(round(p[1]))

    def rasterise(self,
                  img_shape: tuple[int, int],
                  thick_line: int = 2,
                  thick_spot: int = 3,
                  detected_kps: dict[int, tuple[float, float]] | None = None
                  ) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns (mask_lines, mask_goals); both uint8 binaries (0/255).
        """
        h, w = img_shape
        mask_lines = np.zeros((h, w), np.uint8)
        mask_goals = np.zeros((h, w), np.uint8)

        # -------------- straight segments (white + goal frames) -------------
        for A, B in LINE_NAME_SEGMENTS:
            # goal frame? → draw on mask_goals else mask_lines
            is_goal = A.endswith("_POST") and B.endswith("_POST")
            dst = mask_goals if is_goal else mask_lines
            if self.orientation and self.orientation != "none":
                A = self._remap(A, self.orientation)
                B = self._remap(B, self.orientation)
            if A not in TEMPLATE_PT or B not in TEMPLATE_PT:
                continue
            u1, v1 = self._proj(*TEMPLATE_PT[A])
            u2, v2 = self._proj(*TEMPLATE_PT[B])
            cv2.line(dst, (u1, v1), (u2, v2),
                     255, thick_line, cv2.LINE_AA)

        # -------------------- curved segments (centre & arcs) ---------------
        for (x0, y0), (x1, y1) in CHORDS_CURVED:
            u0, v0 = self._proj(x0, y0)
            u1, v1 = self._proj(x1, y1)
            cv2.line(mask_lines, (u0, v0), (u1, v1),
                     255, thick_line, cv2.LINE_AA)

        # ----------------------- spot marks ---------------------------------
        for xs, ys in SPOT_MARKS_NORM:
            us, vs = self._proj(xs, ys)
            cv2.circle(mask_lines, (us, vs),
                       thick_spot, 255, -1, cv2.LINE_AA)

        # ----------------------- goal posts (from keypoints) ----------------
        if detected_kps:
            goal_post_segments = [
                # left goal
                ("L_GOAL_BL_POST", "L_GOAL_TL_POST"),  # left post
                ("L_GOAL_BR_POST", "L_GOAL_TR_POST"),  # right post
                ("L_GOAL_TL_POST", "L_GOAL_TR_POST"),  # cross-bar
                # right goal
                ("R_GOAL_BL_POST", "R_GOAL_TL_POST"),
                ("R_GOAL_BR_POST", "R_GOAL_TR_POST"),
                ("R_GOAL_TL_POST", "R_GOAL_TR_POST"),
            ]
            for A, B in goal_post_segments:
                if self.orientation and self.orientation != "none":
                    A = self._remap(A, self.orientation)
                    B = self._remap(B, self.orientation)

                idx_A = PITCH_POINTS_TO_INTERSECTION.get(A)
                idx_B = PITCH_POINTS_TO_INTERSECTION.get(B)

                if idx_A in detected_kps and idx_B in detected_kps:
                    u1, v1 = (int(c) for c in detected_kps[idx_A])
                    u2, v2 = (int(c) for c in detected_kps[idx_B])
                    cv2.line(mask_goals, (u1, v1), (u2, v2),
                             255, thick_line, cv2.LINE_AA)

        return mask_lines, mask_goals


# ---------------- wrapper (unchanged signature) -----------------------------
def build_line_masks(
    kps: list[tuple[int, float, float, float]],
    image_shape: tuple[int, int],
    conf_th: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    h, w = image_shape
    detected = {
        idx: (x_n * w, y_n * h)
        for idx, x_n, y_n, sc in kps
        if sc >= conf_th
    }
    matcher = TemplateMatcher()
    if not matcher.estimate(detected):
        raise RuntimeError("Homography failed – insufficient inliers.")
    return matcher.rasterise(image_shape, detected_kps=detected)

# -----------------------------------------------------------------------------#
# --------------------------- 3.  Main entry‑point ----------------------------#
# -----------------------------------------------------------------------------#
def main() -> None:
    parser = argparse.ArgumentParser("Key‑point detection + line segmenter")
    parser.add_argument("--frame", default=r"images\frame_0398.jpg", help="input broadcast frame")
    parser.add_argument("--out", default="overlay.png", help="output PNG")
    parser.add_argument("--save_masks", action="store_true",
                        help="additionally save masks as 'lines.png'/'goals.png'")
    args = parser.parse_args()

    # ---- load frame --------------------------------------------------------
    img_bgr = cv2.imread(args.frame)
    if img_bgr is None:
        raise FileNotFoundError(args.frame)
    h_img, w_img = img_bgr.shape[:2]

    # ---- HRNet detection ---------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(device)
    LOGGER.info("Running HRNet inference...")
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    tensor = TRANSFORM(image=rgb)["image"].to(device)
    keypoints = model.get_keypoints(tensor.unsqueeze(0))[0]
    if device == "cuda":
        torch.cuda.synchronize()

    # ---- Line / goal segmentation -----------------------------------------
    LOGGER.info("Estimating homography + rasterising template...")
    mask_lines, mask_goals = build_line_masks(
        keypoints, (h_img, w_img), conf_th=0.1
    )

    # ---- Create overlay for quick QA --------------------------------------
    overlay = img_bgr.copy()
    overlay[mask_lines > 0] = (0, 255, 255)  # yellow for lines
    overlay[mask_goals > 0] = (0, 0, 255)    # red for goals
    cv2.imwrite(args.out, overlay)
    LOGGER.info(f"Overlay saved: {args.out}")

    # ---- Optional: save raw masks -----------------------------------------
    if args.save_masks:
        cv2.imwrite("lines.png", mask_lines)
        cv2.imwrite("goals.png", mask_goals)
        LOGGER.info("Mask images written: lines.png, goals.png")


if __name__ == "__main__":
    main()








