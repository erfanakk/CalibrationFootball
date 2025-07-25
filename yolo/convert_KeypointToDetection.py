import os
import argparse
import shutil

def convert_keypoints_to_detections(label_file, output_file, bbox_w, bbox_h):
    """
    Convert a single YOLO keypoint annotation file to YOLO detection format.
    Each visible keypoint (v > 0) becomes a bounding box of fixed relative size.

    Args:
        label_file (str): Path to the original keypoint .txt file
        output_file (str): Path to write the new detection .txt file
        bbox_w (float): Width of each detection box (relative to image width, between 0 and 1)
        bbox_h (float): Height of each detection box (relative to image height, between 0 and 1)
    """
    with open(label_file, 'r') as f:
        line = f.readline().strip()
        if not line:
            return
    parts = line.split()
    kpt_data = parts[5:]
    num_kpts = len(kpt_data) // 3

    with open(output_file, 'w') as out:
        for i in range(num_kpts):
            x = float(kpt_data[3*i])
            y = float(kpt_data[3*i+1])
            v = int(float(kpt_data[3*i+2]))
            if v > 0:
                cls_id = i
                out.write(f"{cls_id} {x:.6f} {y:.6f} {bbox_w:.6f} {bbox_h:.6f}\n")



def process_labels_dir(labels_dir, output_labels_dir, bbox_w, bbox_h):
    """
    Process all keypoint .txt files in a labels directory and write fixed-size detections.
    """
    os.makedirs(output_labels_dir, exist_ok=True)
    for fname in os.listdir(labels_dir):
        if not fname.endswith('.txt'):
            continue
        in_file = os.path.join(labels_dir, fname)
        out_file = os.path.join(output_labels_dir, fname)
        convert_keypoints_to_detections(in_file, out_file, bbox_w, bbox_h)
    print(f"Converted labels in {labels_dir} -> {output_labels_dir}")


def process_root(root_dir, bbox_w, bbox_h):
    """
    For each split under root_dir (train/valid/test),
    convert its labels folder and optionally copy images folder to output.
    """
    for split in os.listdir(root_dir):
        split_path = os.path.join(root_dir, split)
        if not os.path.isdir(split_path):
            continue
        labels_dir = os.path.join(split_path, 'labels')
        images_dir = os.path.join(split_path, 'images')
        out_labels = os.path.join(split_path, 'labels')


        if os.path.isdir(labels_dir):
            process_labels_dir(labels_dir, out_labels, bbox_w, bbox_h)
        else:
            print(f"Warning: no labels folder at {labels_dir}")




if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Convert YOLO keypoint annotations (per-split images+labels) to fixed-size YOLO detection format."
    )
    parser.add_argument('--root-dir', default=r'football-field-detection.v15i.yolov8', type=str,
                        help='Root directory containing split subdirs (each with images/ and labels/)')
    parser.add_argument('--box-width', type=float, default=0.02,
                        help='Relative width of each detection box (0-1)')
    parser.add_argument('--box-height', type=float, default=0.02,
                        help='Relative height of each detection box (0-1)')
    args = parser.parse_args()

    process_root(args.root_dir, args.box_width, args.box_height)