"""
DuckDuckGo Image Downloader Script

This script downloads images from DuckDuckGo based on a search query. The images are saved in a specified output directory.

Usage:
    python src/download_images.py 

Example:
    To download 5 images of "cats" and save them to the "data/cats" directory:
        python src/download_images.py
        Enter your search query: refrigerator
        Enter the number of images to download: 50
"""

import os
import requests
from urllib.parse import urlparse
from pathlib import Path
from dotenv import load_dotenv
from duckduckgo_search import DDGS

load_dotenv()


def download_images_duckduckgo(query, num_images=10, output_dir="data"):
    """Downloads images from DuckDuckGo based on a query."""

    os.makedirs(os.path.join(output_dir, query), exist_ok=True)

    ddgs = DDGS()
    results = ddgs.images(
        query,
        region="wt-wt",
        safesearch="off",
        timelimit=None,
        size=None,
        color=None,
        type_image="photo",
        layout=None,
        license_image=None,
    )

    image_count = 0
    for image_data in results:
        if image_count >= num_images:
            break

        image_url = image_data.get("image")  # Correctly access the image URL
        if not image_url:
            print("No image URL found in results.")
            continue

        try:
            parsed_url = urlparse(image_url)
            filename = os.path.basename(parsed_url.path) or "image.jpg"
            filename = Path(filename).stem + ".jpg"  # Ensure .jpg extension
            filepath = os.path.join(output_dir, query, filename)
            print(f"Downloading: {image_url} as {filename}")

            image_response = requests.get(image_url, stream=True, timeout=10)
            image_response.raise_for_status()

            with open(filepath, "wb") as f:
                for chunk in image_response.iter_content(1024):
                    f.write(chunk)

            print(f"Downloaded: {image_url} as {filename}")
            image_count += 1

        except requests.exceptions.RequestException as e:
            print(f"Error downloading image: {image_url} - {e}")
        except Exception as e:
            print(f"Error processing image URL: {image_url} - {e}")


if __name__ == "__main__":
    query = input("Enter your search query: ")
    num_images_to_download = int(input("Enter the number of images to download: "))
    download_images_duckduckgo(query, num_images=num_images_to_download)
