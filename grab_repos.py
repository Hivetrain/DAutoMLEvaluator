import os
import bittensor as bt
from huggingface_hub import HfApi
import logging
from typing import Optional
import json
from tqdm import tqdm

CHECKPOINTS_DIR = "./checkpoints"
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def get_repo_from_metadata(subtensor: bt.subtensor, netuid: int, hotkey: str) -> Optional[str]:
    """Retrieves repository information from chain metadata for a given hotkey."""
    try:
        metadata = bt.extrinsics.serving.get_metadata(subtensor, netuid, hotkey)
        if not metadata:
            return None
        
        commitment = metadata["info"]["fields"][0]
        hex_data = commitment[list(commitment.keys())[0]][2:]
        repo = bytes.fromhex(hex_data).decode()
        return repo
    except Exception as e:
        logging.error(f"Failed to retrieve metadata for hotkey {hotkey}: {str(e)}")
        return None

def download_gene_file(repo_name: str, uid: int) -> bool:
    """Downloads best_gene.json from a Hugging Face repository."""
    api = HfApi()
    try:
        # Check if the file exists in the repo
        file_info = api.list_repo_files(repo_id=repo_name)
        if "best_gene.json" not in file_info:
            logging.warning(f"best_gene.json not found in repository {repo_name}")
            return False

        # Get file details to check size
        file_details = [f for f in api.list_repo_tree(repo_id=repo_name) if f.path == "best_gene.json"]
        if not file_details:
            logging.warning(f"Could not get file details for best_gene.json in {repo_name}")
            return False

        # Download the file
        file_path = api.hf_hub_download(repo_id=repo_name, filename="best_gene.json")
        
        # Create new filename based on UID and repo name
        repo_short_name = repo_name.split('/')[-1]
        #repo_username = repo_name.split('/')[-2]
        str_repo_name = repo_name.replace("/", "_")
        new_filename = f"{uid}_{str_repo_name}.json"
        new_filepath = os.path.join(CHECKPOINTS_DIR, new_filename)
        
        # Copy file content to new location
        with open(file_path, 'r') as source, open(new_filepath, 'w') as dest:
            json_content = json.load(source)
            json.dump(json_content, dest, indent=2)
        
        # Remove the temporary download
        os.remove(file_path)
        
        logging.info(f"Successfully downloaded and stored gene from {repo_name} as {new_filename}")
        return True
    
    except Exception as e:
        logging.error(f"Error downloading gene from {repo_name}: {str(e)}")
        return False

def main():
    # Initialize Bittensor subtensor connection
    subtensor = bt.subtensor()
    
    # Specify the subnet UID you want to query
    netuid = 49  # Replace with your desired netuid
    
    # Get the metagraph for the subnet
    metagraph = subtensor.metagraph(netuid)
    
    logging.info(f"Querying subnet {netuid} with {len(metagraph.hotkeys)} miners")
    
    # Track statistics
    total_repos = 0
    successful_downloads = 0
    
    # Process each hotkey in the metagraph
    for uid, hotkey in tqdm(enumerate(metagraph.hotkeys)):
        repo = get_repo_from_metadata(subtensor, netuid, hotkey)
        
        if repo:
            total_repos += 1
            logging.info(f"Found repository for UID {uid}: {repo}")
            
            if download_gene_file(repo, uid):
                successful_downloads += 1
        else:
            logging.warning(f"No repository found for UID {uid} (hotkey: {hotkey})")
    
    # Log summary
    logging.info(f"\nDownload Summary:")
    logging.info(f"Total miners: {len(metagraph.hotkeys)}")
    logging.info(f"Repos found: {total_repos}")
    logging.info(f"Successful downloads: {successful_downloads}")

if __name__ == "__main__":
    main()