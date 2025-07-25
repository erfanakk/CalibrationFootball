# #!/usr/bin/env python
# """
# Key-point-only demo for Eagle with batch processing for improved FPS.
# Usage:
#     python keypoint_inference.py --video_path input.mp4 \
#                                  --out_path keypoints.mp4 \
#                                  --conf 0.30 \
#                                  --batch_size 4
# """
# #keypoint_inference.py


# import cv2
# import torch
# import time
# import logging
# import numpy as np
# import albumentations as A
# from albumentations.pytorch import ToTensorV2
# from pathlib import Path
# from argparse import ArgumentParser
# from collections import deque


# # --- Eagle bits -------------------------------------------------------------
# from eagle.models.keypoint_hrnet import KeypointModel
# from eagle.utils.pitch import INTERSECTION_TO_PITCH_POINTS

# # ---------------------------------------------------------------------------

# def load_model(device: str):
#     """Instantiate HRNet and load pretrained weights."""
#     model = KeypointModel(57).to(device)
#     w_path = Path(r"eagle\models\weights\weights\weights\keypoints_main.pth")
#     model.load_state_dict(torch.load(w_path, map_location=device))
#     model.eval()
#     return model


# def annotate_video(video_path: str,
#                    out_path: str,
#                    conf: float,
#                    show_label: bool,
#                    debug: bool,
#                    batch_size: int = 4):
#     # ---------------- setup -------------------------------------------------
#     log_level = logging.DEBUG if debug else logging.INFO
#     logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s",
#                         level=log_level)
#     logger = logging.getLogger("KP-Demo")

#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     model = load_model(device)
#     logger.info(f"Using device: {device}")
#     logger.info(f"Batch size: {batch_size}")

#     transform = A.Compose([A.Resize(540, 960), A.Normalize(), ToTensorV2()])

#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         raise FileNotFoundError(video_path)

#     native_fps = cap.get(cv2.CAP_PROP_FPS) or 24
#     width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fourcc = cv2.VideoWriter_fourcc(*"mp4v")
#     writer = cv2.VideoWriter(out_path, fourcc, native_fps, (width, height))

#     frame_idx, acc_time = 0, 0.0
#     logger.info("▶ Processing with batch inference...  (press q / Esc to quit preview)")
#     # -----------------------------------------------------------------------

#     # Batch processing variables
#     frame_buffer = deque(maxlen=batch_size)
#     tensor_buffer = []
#     frame_bgr_buffer = []
    
#     while True:
#         # Collect frames for batch processing
#         while len(frame_buffer) < batch_size:
#             grabbed, frame_bgr = cap.read()
#             if not grabbed:
#                 break
            
#             frame_buffer.append(frame_bgr)
#             rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
#             tensor = transform(image=rgb)["image"]
#             tensor_buffer.append(tensor)
#             frame_bgr_buffer.append(frame_bgr)
        
#         # If we don't have enough frames for a full batch, process what we have
#         if len(frame_buffer) == 0:
#             break
            
#         tic = time.perf_counter()

#         # -------- batch key-point inference ---------------------------------
#         batch_tensor = torch.stack(tensor_buffer).to(device)
#         batch_kp_lists = model.get_keypoints(batch_tensor)  # list of list[(idx,x_norm,y_norm,score)]
#         # -------------------------------------------------------------------

#         # -------- process each frame in batch ------------------------------
#         for i, (frame_bgr, kp_list) in enumerate(zip(frame_bgr_buffer, batch_kp_lists)):
#             # Draw points for this frame
#             for idx, x_n, y_n, score in kp_list:
#                 if score < conf:
#                     continue
#                 x = int(x_n * width)
#                 y = int(y_n * height)
#                 cv2.circle(frame_bgr, (x, y), 4, (0, 255, 255), -1)
#                 if show_label:
#                     label = INTERSECTION_TO_PITCH_POINTS[idx]
#                     cv2.putText(frame_bgr, label, (x + 5, y - 5),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.4,
#                                 (255, 255, 255), 1, cv2.LINE_AA)
            
#             # Calculate FPS for this batch
#             toc = time.perf_counter()
#             proc_fps = batch_size / (toc - tic + 1e-9)
            
#             # Overlay FPS
#             cv2.putText(frame_bgr, f"{proc_fps:5.1f} FPS (batch)",
#                         (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
#                         0.7, (0, 255, 0), 2, cv2.LINE_AA)

#             writer.write(frame_bgr)
#             frame_idx += 1
#         # -------------------------------------------------------------------

#         acc_time += toc - tic
        
#         # Clear buffers for next batch
#         frame_buffer.clear()
#         tensor_buffer.clear()
#         frame_bgr_buffer.clear()

#         if debug and frame_idx % (batch_size * 10) == 0:
#             logger.debug(f"Frame {frame_idx:>5d} | "
#                          f"batch FPS: {proc_fps:5.1f}")

#     # ---------------- teardown --------------------------------------------
#     cap.release()
#     writer.release()
#     cv2.destroyAllWindows()
#     if frame_idx:
#         logger.info(f"✔ Saved → {out_path}")
#         logger.info(f"Avg processing FPS: {frame_idx / acc_time:5.2f}")
#         logger.info(f"Processed {frame_idx} frames with batch size {batch_size}")
#     # -----------------------------------------------------------------------


# if __name__ == "__main__":
#     parser = ArgumentParser()
#     parser.add_argument("--video_path", default=r"121364_0.mp4")
#     parser.add_argument("--out_path",  default="annotated_keypoints.mp4")
#     parser.add_argument("--conf", type=float, default=0.30)
#     parser.add_argument("--hide_label", default=True,
#                         help="Do NOT print the point's text label")
#     parser.add_argument("--debug", action="store_true", default=True,
#                         help="Verbose timings & per-frame logs")
#     parser.add_argument("--batch_size", type=int, default=2,
#                         help="Batch size for processing (higher = better FPS, more memory)")
#     args = parser.parse_args()

#     annotate_video(args.video_path,
#                    args.out_path,
#                    conf=args.conf,
#                    show_label=not args.hide_label,
#                    debug=args.debug,
#                    batch_size=args.batch_size)









import cv2
import torch
import time
import logging
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path
from argparse import ArgumentParser
from collections import deque
import torch.profiler # Import profiler

# --- Eagle bits -------------------------------------------------------------
from eagle.models.keypoint_hrnet import KeypointModel
from eagle.utils.pitch import INTERSECTION_TO_PITCH_POINTS

# ---------------------------------------------------------------------------

def load_model(device: str):
    """Instantiate HRNet and load pretrained weights."""
    model = KeypointModel(57).to(device)
    w_path = Path(r"eagle\models\weights\weights\weights\keypoints_main.pth")
    model.load_state_dict(torch.load(w_path, map_location=device))
    model.eval()
    return model


def annotate_video(video_path: str,
                   out_path: str,
                   conf: float,
                   show_label: bool,
                   debug: bool,
                   batch_size: int = 4):
    # ---------------- setup -------------------------------------------------
    log_level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s",
                        level=log_level)
    logger = logging.getLogger("KP-Demo")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(device)
    logger.info(f"Using device: {device}")
    logger.info(f"Batch size: {batch_size}")

    transform = A.Compose([A.Resize(540, 960), A.Normalize(), ToTensorV2()])

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(video_path)

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 24
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, native_fps, (width, height))

    frame_idx, acc_time = 0, 0.0
    logger.info("▶ Processing with batch inference...  (press q / Esc to quit preview)")
    # -----------------------------------------------------------------------

    # Batch processing variables
    frame_buffer = deque(maxlen=batch_size)
    tensor_buffer = []
    frame_bgr_buffer = []
    
    # --- Profiling setup ---
    prof_schedule = torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1)
    # Using 'wait', 'warmup', 'active', 'repeat' to capture a stable window
    # Wait: initial idle time
    # Warmup: run without recording to warm up GPU, etc.
    # Active: actual recording period
    # Repeat: how many times to repeat the cycle
    # You might need to adjust these values based on your video length.
    
    with torch.profiler.profile(
        schedule=prof_schedule,
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/keypoint_inference_profile'),
        with_stack=True,
        profile_memory=True
    ) as prof:
        while True:
            # Collect frames for batch processing
            grabbed_count = 0
            for _ in range(batch_size): # Try to grab up to batch_size frames
                grabbed, frame_bgr = cap.read()
                if not grabbed:
                    break
                frame_buffer.append(frame_bgr)
                rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                tensor = transform(image=rgb)["image"]
                tensor_buffer.append(tensor)
                frame_bgr_buffer.append(frame_bgr)
                grabbed_count += 1
            
            # If no frames were grabbed, exit
            if grabbed_count == 0:
                break
            
            # If we don't have enough frames for a full batch, process what we have
            # (This logic is for the last partial batch)
            if len(frame_buffer) == 0:
                break
                
            tic = time.perf_counter()

            # -------- batch key-point inference ---------------------------------
            batch_tensor = torch.stack(tensor_buffer).to(device)
            batch_kp_lists = model.get_keypoints(batch_tensor)  # list of list[(idx,x_norm,y_norm,score)]
            # -------------------------------------------------------------------

            # -------- process each frame in batch ------------------------------
            for i, (frame_bgr, kp_list) in enumerate(zip(frame_bgr_buffer, batch_kp_lists)):
                # Draw points for this frame
                for idx, x_n, y_n, score in kp_list:
                    if score < conf:
                        continue
                    x = int(x_n * width)
                    y = int(y_n * height)
                    cv2.circle(frame_bgr, (x, y), 4, (0, 255, 255), -1)
                    if show_label:
                        label = INTERSECTION_TO_PITCH_POINTS[idx]
                        cv2.putText(frame_bgr, label, (x + 5, y - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                                    (255, 255, 255), 1, cv2.LINE_AA)
                
                # Overlay FPS (This FPS calculation is per-frame within the batch loop, might be misleading)
                # It's better to calculate FPS for the entire batch processing.
                
                writer.write(frame_bgr)
                frame_idx += 1
            # -------------------------------------------------------------------

            toc = time.perf_counter()
            # Calculate FPS for this batch
            proc_fps = grabbed_count / (toc - tic + 1e-9)
            
            # Overlay FPS on the *last* frame of the batch for display
            if len(frame_bgr_buffer) > 0:
                cv2.putText(frame_bgr_buffer[-1], f"{proc_fps:5.1f} FPS (batch)",
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 0), 2, cv2.LINE_AA)
            
            acc_time += (toc - tic) # Accumulate time for total FPS
            
            # Clear buffers for next batch
            frame_buffer.clear()
            tensor_buffer.clear()
            frame_bgr_buffer.clear()

            if debug and frame_idx % (batch_size * 10) == 0:
                logger.debug(f"Frame {frame_idx:>5d} | "
                             f"batch FPS: {proc_fps:5.1f}")
            
            prof.step() # Tell the profiler to advance one step

    # ---------------- teardown --------------------------------------------
    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    if frame_idx:
        logger.info(f"✔ Saved → {out_path}")
        logger.info(f"Avg processing FPS: {frame_idx / acc_time:5.2f}")
        logger.info(f"Processed {frame_idx} frames with batch size {batch_size}")
    # -----------------------------------------------------------------------


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--video_path", default=r"Provispo-mobile.mp4")
    parser.add_argument("--out_path",  default="annotated_keypoints.mp4")
    parser.add_argument("--conf", type=float, default=0.30)
    parser.add_argument("--hide_label", default=True,
                        help="Do NOT print the point's text label")
    parser.add_argument("--debug", action="store_true", default=True,
                        help="Verbose timings & per-frame logs")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for processing (higher = better FPS, more memory)")
    args = parser.parse_args()

    # Create a directory for logs if it doesn't exist
    Path('./log').mkdir(exist_ok=True)

    annotate_video(args.video_path,
                   args.out_path,
                   conf=args.conf,
                   show_label=not args.hide_label,
                   debug=args.debug,
                   batch_size=args.batch_size)