import os
import shutil
import json
import random
from pathlib import Path
import argparse

def split_coco_dataset(coco_path, train_ratio=0.8):
    coco_path = Path(coco_path)
    images_path = coco_path / 'images'
    result_json_path = coco_path / 'result.json'

    
    if images_path.exists() and result_json_path.exists():
        # Create new folder structure
        new_coco_path = coco_path.parent / 'coco_train_val'
        new_annotations_path = new_coco_path / 'annotations'
        new_train_images_path = new_coco_path / 'train2017'
        new_val_images_path = new_coco_path / 'val2017'
        
        if new_coco_path.exists():
            shutil.rmtree(new_coco_path)

        new_annotations_path.mkdir(parents=True, exist_ok=True)
        new_train_images_path.mkdir(parents=True, exist_ok=True)
        new_val_images_path.mkdir(parents=True, exist_ok=True)
        
        # Load annotations
        with open(result_json_path, 'r') as f:
            annotations = json.load(f)
        
        # Split images
        images = annotations['images']
        random.shuffle(images)
        train_size = int(len(images) * train_ratio)
        train_images = images[:train_size]
        val_images = images[train_size:]
        
        # Save new annotations
        train_annotations = {k: v for k, v in annotations.items() if k != 'images'}
        train_annotations['images'] = train_images
        val_annotations = {k: v for k, v in annotations.items() if k != 'images'}
        val_annotations['images'] = val_images
        
        with open(new_annotations_path / 'instances_train2017.json', 'w') as f:
            json.dump(train_annotations, f)
        with open(new_annotations_path / 'instances_val2017.json', 'w') as f:
            json.dump(val_annotations, f)
        
        # Move images
        for image in train_images:
            shutil.copy(coco_path / image['file_name'], new_train_images_path / image['file_name'].split('/')[-1])
        for image in val_images:
            shutil.copy(coco_path / image['file_name'], new_val_images_path / image['file_name'].split('/')[-1])
        
        print(f"Dataset split into {len(train_images)} train and {len(val_images)} validation images.")
    else:
        print("Dataset doesn't follow the proposed intial folder structure or required files are missing.")

# Example usage
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Split COCO dataset into training and validation sets.")
    parser.add_argument("coco_path", type=str, help="Path to the COCO dataset folder.")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Ratio of training data. Default is 0.8.")
    
    args = parser.parse_args()
    split_coco_dataset(args.coco_path, args.train_ratio)