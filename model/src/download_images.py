"""
Module for downloading images using DuckDuckGo.

This script allows you to download a specified number of images from DuckDuckGo based on a search query and save them to a designated folder.

Usage:
    python download_images.py -q "food in fridge" -n 5 -f "../data/ddgs"

Arguments:
    -q, --query       The search query for DuckDuckGo (required).
    -n, --num_images  The number of images to download (default is 5).
    -f, --folder_path The path where the images will be saved (required).
"""

import argparse
import os
import requests
from duckduckgo_search import DDGS


def download_images(query: str, num_images: int, folder_path: str) -> None:
    """
    Download images from DuckDuckGo using the given query and save them to a specified folder.

    Args:
        query (str): The search query for DuckDuckGo.
        num_images (int): The number of images to download.
        folder_path (str): The path where the images will be saved.

    Returns:
        None
    """
    # Create the folder if it doesn't exist
    if not os.path.exists(folder_path):
        print(f"Creating folder {folder_path}")
        os.makedirs(folder_path)

    # Search for images using DuckDuckGo
    with DDGS() as ddgs:
        results = ddgs.images(query, max_results=num_images)

        for i, result in enumerate(results):
            image_url = result["image"]
            try:
                # Send an HTTP request to the image URL
                response = requests.get(image_url)
                response.raise_for_status()  # Raise an error if the request was not successful

                # Save the image to the folder
                with open(os.path.join(folder_path, f"image_{i + 1}.jpg"), "wb") as f:
                    f.write(response.content)

                print(f"Downloaded image {i + 1}")

            except Exception as e:
                print(f"Could not download image {i + 1}. Error: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download images from DuckDuckGo")
    parser.add_argument(
        "-q", "--query", help="The search query for DuckDuckGo", required=True
    )
    parser.add_argument(
        "-n",
        "--num_images",
        type=int,
        help="The number of images to download",
        default=5,
    )
    parser.add_argument(
        "-f",
        "--folder_path",
        help="The path where the images will be saved",
        required=True,
    )

    args = parser.parse_args()

    download_images(args.query, args.num_images, args.folder_path)
