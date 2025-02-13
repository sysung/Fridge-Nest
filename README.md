# FridgeNest

## Getting Started

### Prerequisites

1. **Python**: Ensure you have Python installed on your system. You can download it from [python.org](https://www.python.org/).

2. **Poetry**: Install Poetry for dependency management and packaging. You can install it using the following command:
    ```sh
    curl -sSL https://install.python-poetry.org | python3 -
    ```

3. **Install Dependencies**: Use Poetry to install the required dependencies:
    ```sh
    poetry install
    ```

### Running the Scripts

1. **Activate Poetry Shell**: Activate the Poetry shell to ensure the environment is set up correctly:
    ```sh
    poetry shell
    ```

2. **Download Images**: Run `download_images.py` to download images based on a search query.
    ```sh
    python src/download_images.py --query refrigerator --num_images 50
    ```

3. **Run Inference**: Run `infer.py` on the downloaded images to perform object detection.
    ```sh
    python src/infer.py --image_dir data/refrigerator --threshold 0.7
    ```

### Label Studio

To start Label Studio:
```bash
docker pull heartexlabs/label-studio:latest
docker run -it -p 8080:8080 -v $(pwd)/mydata:/label-studio/data heartexlabs/label-studio:latest
```

### Difficulties

- Data annotations at scale
- Food hidden behind other foods
- Ambiguous foods (Sauces, Liquids, etc)

