from datasets import Dataset
from pycocotools.coco import COCO

MODEL_NAME = "facebook/detr-resnet-50"
IMAGE_SIZE = 480

coco_root_fp = "model/data/coco/project-1-at-2024-09-15-01-58-839140c6/"
coco_dataset = COCO(coco_root_fp + "result.json")

image_ids = coco_dataset.getImgIds()
print(image_ids)

my_dict = {"a": [1, 2, 3], "b": [4, 5, 6]}
dataset = Dataset.from_dict(my_dict)

print(dataset)