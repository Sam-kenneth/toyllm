import os
import sys
import tensorflow as tf
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)
import config.configs as configs
from src.austen_slm.model import create_model
from src.rag.langchain_wrapper import SherlockRAG
from logger.logging import setup_logging
from exceptions.exception import PipelineException

logger = setup_logging(log_file_path='logs/hub_module.log')

class SherlockHub:
    def __init__(self, mode="base", repo_id=None):
        """
        Modes: 
        'base': Loads initial Austen weights.
        'fine_tuned': Loads Sherlock specialized weights.
        'rag': Uses the LangChain + FAISS pipeline.
        """
        try:
            logger.info(f"Initializing SherlockHub in '{mode}' mode")
            self.mode = mode
            
            if mode == "rag":
                if not repo_id:
                    raise ValueError("repo_id is required for RAG mode")
                
                self.rag_system = SherlockRAG(repo_id=repo_id)
            else:
                
                self.model = create_model()
                self.model(tf.zeros((1, configs.MAX_LEN)))  
                
                
                if mode == "base":
                    weights = configs.SAVE_PATH
                elif mode == "fine_tuned":
                    weights = "data/processed/transformer_sherlock.h5"
                else:
                    raise ValueError(f"Invalid mode: {mode}")
                
                self.model.load_weights(weights)
                logger.info(f"Model loaded from {weights}")
                
        except Exception as e:
            raise PipelineException(f"Failed to initialize SherlockHub in {mode} mode", sys)

    def generate(self,tokenizer=None,prompt="",max_new_tokens=5,temperature=0.5,top_k=10):
        try:
            if self.mode == "rag":
                return self.rag_system.ask(prompt)

            if tokenizer is None:
                raise ValueError("tokenizer is required for base/fine_tuned modes")
            
            input_eval = tokenizer.texts_to_sequences([prompt])
            input_eval = tf.expand_dims(input_eval[0], 0)
            
            generated = []
            for _ in range(max_new_tokens):
                
                predictions = self.model(input_eval, training=False)
                logits = predictions[:, -1, :] / temperature  
                
                
                values, indices = tf.math.top_k(logits, k=top_k)
                
                min_value = tf.reduce_min(values)
                mask = logits < min_value
                logits = tf.where(mask, tf.ones_like(logits) * -1e9, logits)
                
                
                predicted_id = tf.random.categorical(logits, num_samples=1)[0, 0].numpy()
                
                
                input_eval = tf.concat([input_eval, [[predicted_id]]], axis=-1)
                if input_eval.shape[1] > configs.MAX_LEN:
                    input_eval = input_eval[:, 1:]
                    
                generated.append(int(predicted_id))
                if predicted_id == 0: break 
            
            return tokenizer.sequences_to_texts([generated])[0]
        except Exception as e:
            raise PipelineException(f"Generation failed: {str(e)}", sys)