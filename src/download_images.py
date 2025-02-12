import requests
import os
import json

from urllib.parse import urlparse
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Replace with your Unsplash Access Key
UNSPLASH_ACCESS_KEY = os.getenv("UNSPLASH_ACCESS_KEY")

def download_images_unsplash(query, num_images=10, output_dir="data"):
    """Downloads images from Unsplash based on a query."""

    os.makedirs(os.path.join(output_dir, query), exist_ok=True)

    url = "https://api.unsplash.com/search/photos"

    params = {
        "query": query,
        "per_page": num_images,           # Number of images per page (max is 30)
        "page": 1, 
        "client_id": UNSPLASH_ACCESS_KEY  # Your Unsplash API key
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        data = response.json()

        if not data["results"]:
            print("No images found for your query.")
            return

        for image_data in data["results"]:
            image_url = image_data["urls"]["regular"]  # Use "regular", "small", or "full" as needed

            try:
                parsed_url = urlparse(image_url)
                filename = os.path.basename(parsed_url.path) or "image.jpg"
                filename = Path(filename).stem + ".jpg" # Ensure .jpg extension and prevent issues with special characters in name.
                filepath = os.path.join(output_dir, query, filename)

                image_response = requests.get(image_url, stream=True)
                image_response.raise_for_status()

                with open(filepath, 'wb') as f:
                    for chunk in image_response.iter_content(1024):
                        f.write(chunk)

                print(f"Downloaded: {image_url} as {filename}")

            except requests.exceptions.RequestException as e:
                print(f"Error downloading image: {image_url} - {e}")
            except Exception as e:
                print(f"Error processing image URL: {image_url} - {e}")

    except requests.exceptions.RequestException as e:
        print(f"Error searching: {e}")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e} - Response Text: {response.text}")



if __name__ == "__main__":
    query = input("Enter your search query: ")
    num_images_to_download = int(input("Enter the number of images to download: "))
    download_images_unsplash(query, num_images=num_images_to_download)