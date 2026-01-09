import tensorflow as tf
import os
import sys
from typing import Any, List, Optional
from langchain.llms.base import LLM
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)
import config.configs as configs


class SherlockLLM(LLM):
    model: Any
    tokenizer: Any

    @property
    def _llm_type(self) -> str:
        return "custom_sherlock_slm"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        input_ids = self.tokenizer.texts_to_sequences([prompt])[0]
        input_ids = tf.expand_dims(input_ids[-(configs.MAX_LEN-1):], 0)
        
        generated_tokens = []
        curr_input = input_ids
        for _ in range(5): 
            preds = self.model(curr_input, training=False)
            logits = preds[:, -1, :] / 0.7  # Temperature 0.7
        
        
            values, _ = tf.math.top_k(logits, k=5)
            logits = tf.where(logits < values[:, -1:], tf.ones_like(logits) * -1e9, logits)
            next_id = tf.random.categorical(logits, num_samples=1)[0, 0].numpy()
            
            if next_id == 0: break 
            
            generated_tokens.append(int(next_id))
            new_token = tf.cast([[next_id]], tf.float32)
            curr_input = tf.concat([tf.cast(curr_input, tf.float32), new_token], axis=-1)
            if curr_input.shape[1] > configs.MAX_LEN:
                curr_input = curr_input[:, 1:]
            
        return self.tokenizer.sequences_to_texts([generated_tokens])[0]


class SherlockRAG:
    def __init__(self, repo_id: str, index_path: str = "data/processed/faiss_sherlock"):
        from src.hugging_face.hf_downloader import load_custom_slm
        
        
        print(f"Fetching model weights...")
        self.model, self.tokenizer = load_custom_slm(repo_id)
        self.llm = SherlockLLM(model=self.model, tokenizer=self.tokenizer)
        
        
        print(f"Loading knowledge base...")
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vectorstore = FAISS.load_local(
            index_path, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 1})
        )

    def ask(self, query: str) -> str:
        
        result = self.qa_chain.invoke(query)
        return result['result']