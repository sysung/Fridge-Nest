"""
Ultralytics YOLO Inference Script

This script performs object detection on a directory of images using a YOLO model from the ultralytics package.
For each image in the specified input directory, the script runs inference to detect objects, draws bounding
boxes and labels around identified objects, and saves the annotated images to the output directory.

Usage:
    python infer.py --image_dir /path/to/input/images [--weights path/to/model_weights] [--output_dir /path/to/output/directory] [--threshold 0.6]

Arguments:
    --weights
        An optional string specifying the path to the YOLO model weights, or an identifier for built-in weights.
        Default: "yolo11n.pt". If a built-in weights identifier is provided, the weights will be downloaded if
        not found locally.

    --image_dir
        A required string specifying the path to the directory containing the input images. Only files with
        extensions such as .jpg, .jpeg, .png, or .bmp are processed.

    --output_dir
        An optional string defining the directory where the output images will be saved.
        Default: "data/infer". The directory will be created if it does not exist.

    --threshold
        A float defining the confidence threshold for filtering detections.
        Default: 0.6

Example:
    To run the script on images in the "./images" directory using custom weights, a threshold of 0.7, and save the outputs to "./output":
        python src/infer.py --image_dir data/refrigerator --threshold 0.7
"""

import argparse
from pathlib import Path
import cv2
from ultralytics import YOLO
import shutil


def load_model(weights_path):
    """
    Load the YOLO model from the ultralytics package using the given weights path.
    """
    try:
        model = YOLO(weights_path)
    except Exception as e:
        raise RuntimeError(
            "Failed to load YOLO model using ultralytics. Please check the weights path."
        ) from e
    return model


def run_inference(model, image_path, output_path, threshold):
    """
    Performs inference on the input image using the ultralytics model.
    """
    # Read the image for display and drawing results
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image not found: {image_path}")

    # Run inference; ultralytics accepts image paths directly
    results = model(image_path)
    # Remove detections for bottles, bowls, and refrigerators, and those with confidence below the threshold
    result = results[0]
    valid_indices = []

    # Loop over each detection's box by index
    for i, detection in enumerate(result.boxes):
        label = result.names.get(int(detection.cls.numpy().item()))
        if label not in ['bottle', 'bowl', 'refrigerator'] and detection.conf > threshold:
            valid_indices.append(i)

    if not valid_indices:
        print(f"No valid detections for image {image_path}.")
        return

    # Create a new results object by filtering with valid indices
    result = result[valid_indices]

    # Retrieve detections from result boxes
    boxes = result.boxes
    xyxy = boxes.xyxy.cpu().numpy()  # bounding boxes [x1, y1, x2, y2]
    confs = boxes.conf.cpu().numpy()  # confidence scores
    clss = boxes.cls.cpu().numpy()  # class ids

    for (x1, y1, x2, y2), conf, cls in zip(xyxy, confs, clss):
        # Draw bounding box
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        label_text = f"{result.names.get(int(cls), 'Unknown')}:{conf:.2f}"

        # Determine text size for the label
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 2
        (w, h), _ = cv2.getTextSize(label_text, font, font_scale, thickness)

        # Set position for label background and text
        text_x = int(x1)
        text_y = int(y1) - 10
        # If the text goes above the image, adjust it
        if text_y - h - 4 < 0:
            text_y = int(y1) + h + 10

        # Coordinates for the background rectangle
        background_top_left = (text_x, text_y - h - 4)
        background_bottom_right = (text_x + w, text_y + 4)
        # Draw filled rectangle for text background (green)
        cv2.rectangle(
            img, background_top_left, background_bottom_right, (0, 255, 0), -1
        )

        # Draw text (in black for contrast)
        cv2.putText(
            img, label_text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness
        )

    cv2.imwrite(output_path, img)
    print(f"Inference complete. Output saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Ultralytics YOLO Inference Script for a directory of images"
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="yolo11n.pt",
        help="Identifier for YOLO built-in weights (downloads if not found locally)",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        required=True,
        help="Path to the directory containing input images",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/infer",
        help="Directory to save the output images",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.6,
        help="Confidence threshold for filtering detections",
    )
    args = parser.parse_args()

    input_dir = Path(args.image_dir)
    if not input_dir.is_dir():
        raise ValueError(f"Not a valid directory: {args.image_dir}")

    output_dir = Path(args.output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(args.weights)

    # Process all images in the input directory
    for image_path in input_dir.glob("*"):
        if image_path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".bmp"]:
            continue
        output_filename = image_path.stem + "_out" + image_path.suffix
        output_path = output_dir / output_filename
        run_inference(model, str(image_path), str(output_path), args.threshold)


if __name__ == "__main__":
    main()
