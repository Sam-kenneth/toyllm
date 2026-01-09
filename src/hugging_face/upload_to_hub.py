import os
import sys
from huggingface_hub import HfApi
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)
import config.configs as configs
from logger.logging import setup_logging
from exceptions.exception import PipelineException

logger = setup_logging(log_file_path='logs/upload_to_hub.log')

def upload_model_to_hub(repo_id, weights_path=None, tokenizer_path=None):
    try:
        logger.info(f"Uploading to {repo_id}")
        
        
        if weights_path is None:
            weights_path = configs.FINE_TUNED_PATH
        if tokenizer_path is None:
            tokenizer_path = "data/processed/tokenizer.json"
        
        
        api = HfApi()
        api.upload_file(
            path_or_fileobj=weights_path,
            path_in_repo="data/processed/transformer_sherlock.h5",
            repo_id=repo_id
        )
        api.upload_file(
            path_or_fileobj=tokenizer_path,
            path_in_repo="tokenizer.json",
            repo_id=repo_id
        )
        
        logger.info(f"Upload complete: https://huggingface.co/{repo_id}")
        
    except Exception as e:
        raise PipelineException(f"Failed to upload to {repo_id}", sys)


if __name__ == "__main__":
    repo_id = configs.REPO_ID
    upload_model_to_hub(repo_id)