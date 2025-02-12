import mlflow
import os

from dotenv import load_dotenv
from ultralytics import YOLO


def main():
    load_dotenv()

    uri = os.getenv("MLFLOW_URI")
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment("fridge-nest-food-detection")

    with mlflow.start_run() as run:
        model = YOLO("yolo11n.pt")


if __name__ == "__main__":
    main()
