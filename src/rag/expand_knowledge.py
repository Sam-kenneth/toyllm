import sys
import requests
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from logger.logging import setup_logging
from exceptions.exception import PipelineException

logger = setup_logging(log_file_path='logs/expand_knowledge.log')


SHERLOCK_BOOKS = {
    "The_Hound_of_the_Baskervilles": "2852",
    "The_Return_of_Sherlock_Holmes": "108",
    "The_Sign_of_the_Four": "2097",
    "A_Study_in_Scarlet": "2850",
    "His_Last_Bow": "2347"
}

def download_extra_sherlock():
    try:
        logger.info("Starting Sherlock corpus download")
        
        os.makedirs("data/raw", exist_ok=True)

        
        logger.info("Downloading Adventures of Sherlock Holmes (primary)...")
        
        primary_url = "https://www.gutenberg.org/cache/epub/1661/pg1661.txt"
        primary_path = "data/raw/doyle-sherlock.txt"
        
        resp = requests.get(primary_url)
        if resp.status_code == 200:
            with open(primary_path, "w", encoding="utf-8") as f:
                f.write(resp.text)
            logger.info(f"Successfully saved primary file to {primary_path}")
        else:
            logger.error("Failed to download primary Sherlock file")
        
        combined_path = "data/raw/sherlock_mega_corpus.txt"
        
        with open(combined_path, "w", encoding="utf-8") as full_file:
            for title, book_id in SHERLOCK_BOOKS.items():
                logger.info(f"Downloading {title}...")
                url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
                response = requests.get(url)
                if response.status_code == 200:
                    text = response.text
                    full_file.write(text + "\n\n")
                    logger.info(f"Downloaded {title}")
                else:
                    logger.error(f"Failed to download {title}")
                    
        logger.info(f"Mega Corpus created at {combined_path}")
        
    except Exception as e:
        raise PipelineException("Failed to download Sherlock corpus", sys)

if __name__ == "__main__":
    download_extra_sherlock()