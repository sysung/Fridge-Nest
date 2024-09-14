"""
This script processes an input image to detect objects using a pre-trained DETR model, filters the detections based on a confidence score threshold, and plots the results.

Usage:
    python detr_resnet_50.py <image_path> [--threshold 0.7]

Arguments:
    <image_path>      The path to the image file (required).
    --threshold       The confidence score threshold for filtering detections (default is 0.7).

Example:
    python detr_resnet_50.py image.jpg --threshold 0.7
"""

import argparse
from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch

def load_model_and_processor():
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
    model.eval()
    return processor, model

def predict(processor, model, image, threshold):
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes)[0]
    keep = results["scores"] > threshold
    scores = results["scores"][keep]
    boxes = results["boxes"][keep]
    labels = results["labels"][keep]
    return scores, boxes, labels

def plot_results(pil_img, scores, boxes, labels, model):
    """
    Plot the detection results on the image.

    Args:
        pil_img (PIL.Image.Image): The image to plot on.
        scores (torch.Tensor): The detection scores.
        boxes (torch.Tensor): The bounding boxes.
        labels (torch.Tensor): The detection labels.
        model (DetrForObjectDetection): The DETR model to access label names.
    """
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = np.random.rand(len(scores), 3)
    for score, label, (xmin, ymin, xmax, ymax), color in zip(scores, labels, boxes, colors):
        ax.add_patch(
            plt.Rectangle(
                (xmin, ymin),
                xmax - xmin,
                ymax - ymin,
                fill=False,
                color=color,
                linewidth=3,
            )
        )
        label_name = model.config.id2label[label.item()]
        text = f"{label_name}: {score:.2f}"
        ax.text(xmin, ymin, text, fontsize=7, bbox=dict(facecolor="yellow", alpha=0.5))
        box = [round(i.item(), 2) for i in [xmin, ymin, xmax, ymax]]
        print(
            f"Detected {label_name} with confidence "
            f"{round(score.item(), 3)} at location {box}"
        )
    plt.axis("off")
    plt.show()

def main(image_path, threshold=0.7):
    """
    Main function to perform object detection and plot the results.

    Args:
        image_path (str): The path to the image file.
        threshold (float): The confidence score threshold.
    """
    processor, model = load_model_and_processor()
    image = Image.open(image_path)
    scores, boxes, labels = predict(processor, model, image, threshold)
    plot_results(image, scores, boxes, labels, model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DETR object detection on an image.")
    parser.add_argument("image_path", type=str, help="The path to the image file.")
    parser.add_argument("--threshold", type=float, default=0.7, help="The confidence score threshold.")
    args = parser.parse_args()

    main(args.image_path, args.threshold)