import json
import os
import hashlib
import time
import argparse
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
from io import BytesIO


def generate_hash(s):
    """Generate SHA-256 hash."""
    return hashlib.sha256(s.encode('utf-8')).hexdigest()


def save_images(dataset_path, split, image_root, annotation_root, limit_per_folder=500000):
    """
    Load the HuggingFace dataset and save images locally, organizing them into subfolders with a specified limit.

    Args:
        dataset_path (str): Path to the dataset.
        split (str): Dataset split (e.g., 'train', 'test').
        image_root (str): Root directory to save images.
        annotation_root (str): Root directory to save json files.
        limit_per_folder (int): Maximum number of images per subfolder.
    """
    # Load dataset with streaming to handle large datasets efficiently
    dataset = load_dataset(dataset_path, split=split, streaming=True)

    # Create root directory if it doesn't exist
    os.makedirs(image_root, exist_ok=True)

    folder_index = 0
    image_count = 0
    item_count = 0
    sub_folder = f'subfolder_{folder_index}'
    current_folder = os.path.join(image_root, sub_folder)
    os.makedirs(current_folder, exist_ok=True)
    os.makedirs(annotation_root, exist_ok=True)

    print(f"Starting to save images to {image_root}, with up to {limit_per_folder} images per subfolder.")

    output_data = []
    # Iterate over the dataset with a progress bar
    for item in tqdm(dataset, desc="Saving images"):
        if image_count >= limit_per_folder:
            # Save output data for the previous folder to a JSON file
            json_file = os.path.join(annotation_root, f'emova_part{folder_index}.json')
            with open(json_file, 'w') as f:
                json.dump(output_data, f, indent=4)
            print(f"Saved output data to {json_file}")
            output_data = []

            folder_index += 1
            sub_folder = f'subfolder_{folder_index}'
            current_folder = os.path.join(image_root, sub_folder)
            os.makedirs(current_folder, exist_ok=True)
            image_count = 0

        # Generate a unique filename using the item ID and a hash of the current time
        current_time = time.strftime('%Y%m%d%H%M%S', time.localtime())
        hash_str = generate_hash(current_time)
        filename = f"{item['id']}_{hash_str[-8:]}.png"
        image_path = os.path.join(current_folder, filename)

        item_count += 1
        output_data.append(item)

        if 'image' not in item:  # prue text conversations.
            continue

        # Retrieve and save the image
        image = item.pop('image')
        # try:
        if isinstance(image, Image.Image):
            image.save(image_path)
        else:
            # Convert to PIL Image if necessary
            image_stream = BytesIO(image['bytes'])
            image = Image.open(image_stream)
            image.save(image_path)
        # except Exception as e:
        #     print(f"Failed to save image {filename}: {e}")
        #     continue

        item['image'] = os.path.join(sub_folder, filename)

        image_count += 1

    # After finishing the loop, save any remaining output_data
    if output_data:
        json_file = os.path.join(annotation_root, f'emova_part{folder_index}.json')
        with open(json_file, 'w') as f:
            json.dump(output_data, f, indent=4)
        print(f"Saved output data to {json_file}")

    print("All images have been successfully saved.")


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Save HuggingFace dataset images to local directories.")

    parser.add_argument(
        '--dataset_path',
        type=str,
        required=True,
        help="Path to the HuggingFace dataset (e.g., 'lmms-lab/POPE')."
    )

    parser.add_argument(
        '--split',
        type=str,
        default='test',
        choices=['train', 'test'],
        help="Dataset split to use (default: 'test')."
    )

    parser.add_argument(
        '--image_root',
        type=str,
        required=True,
        help="Root directory to save images."
    )

    parser.add_argument(
        '--annotation_root',
        type=str,
        required=True,
        help="Root directory to save images."
    )

    parser.add_argument(
        '--limit_per_folder',
        type=int,
        default=500000,
        help="Maximum number of images per subfolder (default: 500000)."
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    save_images(
        dataset_path=args.dataset_path,
        split=args.split,
        image_root=args.image_root,
        annotation_root=args.annotation_root,
        limit_per_folder=args.limit_per_folder
    )
