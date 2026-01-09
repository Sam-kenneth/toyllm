import os
import sys
import json
import tensorflow as tf
from huggingface_hub import hf_hub_download
from tensorflow.keras.preprocessing.text import tokenizer_from_json
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)
import config.configs as configs
from src.austen_slm.model import create_model
from logger.logging import setup_logging
from exceptions.exception import PipelineException

logger = setup_logging(log_file_path='logs/hf_downloader.log')

def load_custom_slm(repo_id):
    try:
        logger.info(f"Loading model from {repo_id}")
        
        
        weights_path = hf_hub_download(repo_id=repo_id, filename="data/processed/transformer_sherlock.h5")
        tokenizer_path = hf_hub_download(repo_id=repo_id, filename="tokenizer.json")
        
        
        with open(tokenizer_path, 'r') as f:
            tokenizer_data = json.load(f)
            tokenizer = tokenizer_from_json(tokenizer_data)
        
        
        model = create_model()
        model(tf.zeros((1, configs.MAX_LEN)))
        model.load_weights(weights_path)
        
        logger.info("Model loaded successfully")
        return model, tokenizer
        
    except Exception as e:
        raise PipelineException(f"Failed to load model from {repo_id}", sys)


if __name__ == "__main__":
    repo_id = configs.REPO_ID
    model, tokenizer = load_custom_slm(repo_id)