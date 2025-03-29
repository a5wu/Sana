`# Modified again to include height, width, and use 'prompt' key in JSON
import os
import pyarrow.parquet as pq
import webdataset as wds
import io
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
import json
import argparse
import glob
from collections import defaultdict

def get_image_bytes_validate_and_dims(img_bytes, filename_for_error):
    """
    Tries to open image bytes with PIL to validate, get format, and dimensions.

    Returns:
        tuple: (validated_bytes, img_format, width, height) or (None, None, None, None) on error.
    """
    try:
        img = Image.open(io.BytesIO(img_bytes))
        img.verify() # Verify core image data integrity

        # Reopen after verify to access properties
        img = Image.open(io.BytesIO(img_bytes))
        width, height = img.size
        img_format = img.format.lower() if img.format else 'jpeg' # Default to jpeg if format is missing

        if img_format not in ['jpeg', 'png', 'webp']: # Added webp as common format
             print(f"Warning: Unexpected image format '{img_format}' for {filename_for_error}. Attempting to save as jpeg.")
             img_format = 'jpeg' # Default to saving as jpeg

        # Ensure image is RGB for JPEG saving, handle PNG transparency if needed
        if img_format == 'jpeg' and img.mode != 'RGB':
            print(f"Converting {filename_for_error} from {img.mode} to RGB for JPEG.")
            img = img.convert('RGB')
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG')
            validated_bytes = buffer.getvalue()
        elif img_format == 'png' and img.mode == 'RGBA':
             # Decide how to handle transparency - convert to RGB or keep RGBA
             # Converting to RGB is safer if VAE expects 3 channels
             # print(f"Converting RGBA PNG {filename_for_error} to RGB.")
             # img = img.convert('RGB')
             # buffer = io.BytesIO()
             # img.save(buffer, format='PNG')
             # validated_bytes = buffer.getvalue()
             validated_bytes = img_bytes # Keep original RGBA bytes for now
        else:
            validated_bytes = img_bytes # Keep original bytes if already suitable

        return validated_bytes, img_format, width, height
    except UnidentifiedImageError:
        print(f"Error: Cannot identify image file {filename_for_error}. Skipping.")
        return None, None, None, None
    except Exception as e:
        print(f"Error validating/processing image {filename_for_error}: {e}")
        return None, None, None, None

def convert_parquet_to_webdataset_with_dims(parquet_files, output_dir, output_prefix, samples_per_shard=10000):
    """
    Converts Parquet files to WebDataset TAR shards.
    Includes 'prompt', 'width', and 'height' in the JSON metadata.
    Keeps only the longest caption as the 'prompt'.

    Args:
        parquet_files (list): List of paths to input Parquet files.
        output_dir (str): Directory to save the output TAR shards.
        output_prefix (str): Prefix for the output shard names (e.g., 'coco-train').
        samples_per_shard (int): Maximum number of samples per TAR shard.
    """
    os.makedirs(output_dir, exist_ok=True)
    shard_pattern = os.path.join(output_dir, f"{output_prefix}-%06d.tar")
    total_images_processed = 0

    with wds.ShardWriter(shard_pattern, maxcount=samples_per_shard) as sink:
        # Store the longest prompt text found so far for each image
        image_data = defaultdict(lambda: {"prompt_text": "", "image_bytes": None})

        print("Step 1: Reading Parquet files and finding longest prompts...")
        # --- TESTING: Limit files and rows ---
        # parquet_files = parquet_files[:1] # Process only the first parquet file
        # print(f"--- TESTING: Limiting to first parquet file: {parquet_files[0]} ---")
        # --- End Testing ---

        for pf_path in tqdm(parquet_files, desc="Reading Parquets"):
            try:
                table = pq.read_table(pf_path)
                df = table.to_pandas()

                # --- TESTING: Limit rows ---
                # print(f"--- TESTING: Limiting to first 10 rows of {os.path.basename(pf_path)} ---")
                # df = df.head(10)
                 # --- End Testing ---

                for index, row in df.iterrows():
                    filename = row['filename']
                    caption = row['caption']
                    image_field = row['image'] # Could be bytes directly or dict {'bytes': ...}

                    # Extract image bytes carefully
                    if isinstance(image_field, dict) and 'bytes' in image_field:
                        image_bytes = image_field['bytes']
                    elif isinstance(image_field, bytes):
                        image_bytes = image_field
                    else:
                        print(f"Warning: Image data for {filename} is not bytes or expected dict. Type: {type(image_field)}. Skipping.")
                        continue

                    # Store image bytes once, update prompt if current one is longer
                    if image_data[filename]["image_bytes"] is None:
                        image_data[filename]["image_bytes"] = image_bytes
                        image_data[filename]["prompt_text"] = caption
                    elif len(caption) > len(image_data[filename]["prompt_text"]):
                        image_data[filename]["prompt_text"] = caption

            except Exception as e:
                print(f"Error reading or processing Parquet file {pf_path}: {e}")
                continue

        print(f"\nStep 2: Writing {len(image_data)} unique images to WebDataset shards...")
        for base_filename, data in tqdm(image_data.items(), desc="Writing Shards"):
            image_bytes = data["image_bytes"]
            prompt_text = data["prompt_text"] # Longest prompt text found

            if image_bytes is None or not prompt_text:
                print(f"Warning: Missing image bytes or prompt for {base_filename}. Skipping.")
                continue

            # Validate image, get format and dimensions
            validated_bytes, img_format, width, height = get_image_bytes_validate_and_dims(image_bytes, base_filename)

            if validated_bytes is None:
                continue # Skip if image is invalid

            # Create the final sample dictionary for WebDataset
            sample = {
                "__key__": os.path.splitext(base_filename)[0], # Base filename as key
                img_format: validated_bytes,                   # Image bytes with format extension as key
                "json": {                                      # JSON metadata
                    "prompt": prompt_text,                     # The selected prompt text
                    "width": width,                            # Image width
                    "height": height                           # Image height
                    # Optional: add cocoid back if needed: "cocoid": data.get("cocoid", None)
                }
            }

            # Write the sample
            sink.write(sample)
            total_images_processed += 1

    print(f"\nFinished conversion.")
    print(f"Processed {total_images_processed} unique images.")
    print(f"WebDataset shards saved in: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert COCO Parquet shards to WebDataset TAR shards with height, width, and prompt in JSON.")
    parser.add_argument("--input_dir", required=True, help="Directory containing the Parquet files (e.g., 'coco_captions/data').")
    parser.add_argument("--output_dir", required=True, help="Directory where the output WebDataset TAR shards will be saved.")
    parser.add_argument("--split", required=True, choices=['train', 'test'], help="Which split to process ('train' or 'test').")
    parser.add_argument("--samples_per_shard", type=int, default=10000, help="Number of image samples per TAR shard.")

    args = parser.parse_args()

    parquet_pattern = os.path.join(args.input_dir, f"{args.split}-*.parquet")
    parquet_files = sorted(glob.glob(parquet_pattern)) # Sort for consistent processing order

    if not parquet_files:
        print(f"Error: No Parquet files found matching pattern '{parquet_pattern}'")
    else:
        print(f"Found {len(parquet_files)} Parquet files for split '{args.split}'.")
        output_shard_dir = os.path.join(args.output_dir, args.split)
        output_prefix = f"coco-{args.split}"
        # Call the updated function
        convert_parquet_to_webdataset_with_dims(parquet_files, output_shard_dir, output_prefix, args.samples_per_shard)