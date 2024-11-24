import os
import bittensor as bt
from huggingface_hub import HfApi
import json
from tqdm import tqdm
import torch

def download_checkpoints(netuid=49, save_dir="./checkpoints", network = "finney"):
    """
    Download checkpoints from specified subnet miners.

    Args:
        netuid (int): The subnet ID to query (default: 49)
        save_dir (str): Directory to save checkpoint files (default: './checkpoints')

    Returns:
        dict: Summary of the download process
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Initialize Bittensor and HuggingFace connections
    subtensor = bt.subtensor(network=network)
    hf_api = HfApi()

    # Get the metagraph for the subnet
    metagraph = subtensor.metagraph(netuid, lite=False)

    print(f"Found {len(metagraph.hotkeys)} miners in subnet {netuid}")

    weights = metagraph.W[45]

    top_indices = weights.argsort()[-5:][::-1].tolist()
    print(top_indices)
    # Filter hotkeys to only include top N miners
    selected_hotkeys = [metagraph.hotkeys[i] for i in top_indices]

    stats = {"total_miners": len(metagraph.hotkeys), "repos_found": 0, "successful_downloads": 0}

    # Process each hotkey
    for uid, hotkey in tqdm(enumerate(metagraph.hotkeys), desc="Processing miners"):
        # Get repository from metadata
        if uid not in top_indices:
            continue

        try:
            metadata = bt.extrinsics.serving.get_metadata(subtensor, netuid, hotkey)
            if not metadata:
                continue

            commitment = metadata["info"]["fields"][0]
            hex_data = commitment[list(commitment.keys())[0]][2:]
            try:
                repo = bytes.fromhex(hex_data).decode().split(":")[0]
            except:
                continue

            stats["repos_found"] += 1
            print(f"Found repository for UID {uid}: {repo}")

            # Download gene file
            try:
                # Check if file exists
                if "best_gene.json" not in hf_api.list_repo_files(repo_id=repo):
                    print(f"No best_gene.json in {repo}")
                    continue

                # Download and save file
                file_path = hf_api.hf_hub_download(repo_id=repo, filename="best_gene.json")

                # Create new filename
                str_repo_name = repo.replace("/", "_")
                new_filename = f"{uid}_{str_repo_name}.json"
                new_filepath = os.path.join(save_dir, new_filename)

                # Save with proper formatting
                with open(file_path, 'r') as source, open(new_filepath, 'w') as dest:
                    json_content = json.load(source)
                    json.dump(json_content, dest, indent=2)

                # Cleanup temporary file
                os.remove(file_path)

                stats["successful_downloads"] += 1
                print(f"Successfully downloaded gene from {repo}")

            except Exception as e:
                print(f"Error downloading from {repo}: {str(e)}")

        except Exception as e:
            print(f"Error processing hotkey {hotkey}: {str(e)}")

    # Print summary
    print("\nDownload Summary:")
    print(f"Total miners: {stats['total_miners']}")
    print(f"Repos found: {stats['repos_found']}")
    print(f"Successful downloads: {stats['successful_downloads']}")

    return stats

if __name__ == "__main__":
    download_checkpoints(netuid=49, save_dir="./my_checkpoints")