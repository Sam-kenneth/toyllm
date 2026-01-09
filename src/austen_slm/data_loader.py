import sys
import os
import nltk
import numpy as np
from nltk.corpus import gutenberg
from tensorflow.keras.preprocessing.text import Tokenizer
from logger.logging import setup_logging
from exceptions.exception import PipelineException

logger = setup_logging(log_file_path='logs/data_loader.log')

def prepare_data(vocab_size=10000, seq_length=50, corpus_type='austen'):
            
    try:
        logger.info(f"Loading {corpus_type} corpus")
        
        nltk.download('gutenberg', quiet=True)
        
        if corpus_type == 'austen':
            files = ['austen-emma.txt', 'austen-persuasion.txt', 'austen-sense.txt']
    
            raw_texts = [gutenberg.raw(f).lower().replace("\n", " ") for f in files]
    
        elif corpus_type == 'sherlock':
    
            sherlock_path = "data/raw/doyle-sherlock.txt"
    
            if not os.path.exists(sherlock_path):
                logger.error(f"Mega corpus not found at {sherlock_path}. Run expand_knowledge.py first.")
                raise FileNotFoundError(f"Missing local corpus: {sherlock_path}")
    
    
            with open(sherlock_path, "r", encoding="utf-8") as f:
        
                raw_texts = [f.read().lower().replace("\n", " ")]
        
        
        tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
        tokenizer.fit_on_texts(raw_texts)
        
        
        sequences = []
        window_size = seq_length + 1 
        
        for text in raw_texts:
            token_list = tokenizer.texts_to_sequences([text])[0]
            for i in range(window_size, len(token_list)):
                seq = token_list[i-window_size:i]
                sequences.append(seq)
                
        sequences = np.array(sequences)
        logger.info(f"Data prepared: {sequences.shape[0]} sequences")
        
        return sequences[:, :-1], sequences[:, 1:], tokenizer
        
    except Exception as e:
        raise PipelineException(f"Failed to prepare {corpus_type} data", sys)