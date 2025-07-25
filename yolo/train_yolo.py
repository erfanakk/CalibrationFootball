import argparse
from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser(description='Train YOLOv8 model without augmentation')
    parser.add_argument('--data', type=str, default=r"dataset.yaml",
                        help='Path to data.yaml file')
    parser.add_argument('--model', type=str, default='yolo11n.pt',
                        help='Pretrained YOLOv8 model or path (e.g., yolo11n.pt)')
    parser.add_argument('--epochs', type=int, default=80,
                        help='Number of training epochs')
    parser.add_argument('--imgsz', type=int, default=736,
                        help='Image size for training (square)')
    parser.add_argument('--batch', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--device', type=str, default='0',
                        help='CUDA device, e.g., "0" or "cpu"')
    parser.add_argument('--project', type=str, default='run',
                        help='Directory to store training results')
    parser.add_argument('--name', type=str, default='keypoint',
                        help='Name of this training run')
    parser.add_argument('--exist-ok', action='store_true',
                        help='Overwrite existing run if it exists')
    return parser.parse_args()

def main():
    args = parse_args()

    # Initialize model
    model = YOLO(args.model)

    # Train with augmentation disabled
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        exist_ok=args.exist_ok,
        augment=False,        # disable all augmentations
        mosaic=False,         # disable mosaic
        mixup=False,          # disable mixup
        copy_paste=False,      # disable copy-paste
        cls=2,     # Weight of the classification loss
        degrees=5,
        fliplr=0,
    )

if __name__ == '__main__':
    main()