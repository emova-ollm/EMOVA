import json
import os
import hashlib
import time
import argparse
from pathlib import Path
from datasets import load_dataset, get_dataset_config_names
from PIL import Image
from tqdm import tqdm
from io import BytesIO


def save_images(dataset_path, config, split, save_root):
    """
    Load the HuggingFace dataset and save images locally, organizing them into subfolders with a specified limit.

    Args:
        dataset_path (str): Path to the dataset.
        name (str): Dataset subset config name.
        split (str): Dataset split (e.g., 'train', 'test').
        save_root (str): Root directory to save datasets.
    """
    # prepare arguments
    config = get_dataset_config_names(dataset_path) if config == 'all' else [config]
    config.sort()
    save_root = os.path.join(save_root, dataset_path.rstrip('/').split('/')[-1])
    annotation_root = os.path.join(save_root, 'annotations')
    Path(annotation_root).mkdir(parents=True, exist_ok=True)
    
    for each_config in tqdm(config, desc="Saving dataset: {}".format(dataset_path)):
        # Load dataset with streaming to handle large datasets efficiently
        dataset = load_dataset(dataset_path, name=each_config, split=split, streaming=True)
        
        # Create root directory if it doesn't exist
        image_root = os.path.join(save_root, each_config)
        os.makedirs(image_root, exist_ok=True)
        
        # Iterate over the dataset with a progress bar
        output_data = []
        for index, item in enumerate(tqdm(dataset, desc="Saving images: {}".format(each_config))):
            if 'image' in item:
                image_path = os.path.join(image_root, "{}.png".format(index))
                image = item.pop('image')
                if isinstance(image, Image.Image):
                    image.save(image_path)
                else:
                    # Convert to PIL Image if necessary
                    image_stream = BytesIO(image['bytes'])
                    image = Image.open(image_stream)
                    image.save(image_path)
                item['image'] = os.path.join(each_config, "{}.png".format(index))
            output_data.append(item)

        # Save output data for the previous folder to a JSON file
        json_file = os.path.join(annotation_root, '{}.json'.format(each_config))
        with open(json_file, 'w') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=4)
            print(f"Saved output data to {json_file}")


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Save HuggingFace dataset images to local directories.")

    parser.add_argument(
        '--dataset_path',
        type=str,
        required=True,
        help="Path to the HuggingFace dataset (e.g., 'Emova-ollm/emova-sft-4m')."
    )

    parser.add_argument(
        '--config',
        type=str,
        default='all',
        help="Dataset subset config name (default: 'all' means downloading all the subsets)."
    )

    parser.add_argument(
        '--split',
        type=str,
        default='train',
        choices=['train', 'test'],
        help="Dataset split to use (default: 'train')."
    )

    parser.add_argument(
        '--save_root',
        type=str,
        default='./data',
        help="Root directory to save datasets."
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    print(args)

    save_images(
        dataset_path=args.dataset_path,
        config=args.config,
        split=args.split,
        save_root=args.save_root
    )
