import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


VOCAB_SIZE = 10000
MAX_LEN = 50
D_MODEL = 256
NUM_LAYERS = 4
NUM_HEADS = 8
D_FF = 1024
DROPOUT_RATE = 0.1


AUSTEN_WEIGHTS = "data/processed/decoder_only_weights.h5"
SHERLOCK_WEIGHTS = "data/processed/transformer_sherlock.h5"
TOKENIZER_PATH = "data/processed/tokenizer.json"
FAISS_INDEX_DIR = "data/processed"



class PositionalEncoding(layers.Layer):
    def __init__(self, max_len, d_model):
        super().__init__()
        pos = np.arange(max_len)[:, np.newaxis]
        i = np.arange(d_model)[np.newaxis, :]
        angle_rates = 1.0 / np.power(10000.0, (2 * (i // 2)) / d_model)
        angles = pos * angle_rates
        pe = np.zeros(angles.shape)
        pe[:, 0::2] = np.sin(angles[:, 0::2])
        pe[:, 1::2] = np.cos(angles[:, 1::2])
        self.pe = tf.constant(pe[np.newaxis, ...], dtype=tf.float32)

    def call(self, x):
        seq_len = tf.shape(x)[1]
        return x + tf.cast(self.pe[:, :seq_len, :], dtype=x.dtype)

class DecoderBlock(layers.Layer):
    def __init__(self, d_model, num_heads, d_ff, dropout_rate=0.1):
        super().__init__()
        self.mha = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.ffn = tf.keras.Sequential([
            layers.Dense(d_ff, activation="relu"),
            layers.Dense(d_model),
        ])
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)

    def call(self, x, mask=None, training=False):
        attn_out = self.mha(x, x, x, attention_mask=mask, training=training)
        out1 = self.norm1(x + self.dropout1(attn_out, training=training))
        ffn_out = self.ffn(out1)
        return self.norm2(out1 + self.dropout2(ffn_out, training=training))

class TransformerModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.embedding = layers.Embedding(VOCAB_SIZE, D_MODEL)
        self.pos_encoding = PositionalEncoding(MAX_LEN, D_MODEL)
        self.dec_layers = [
            DecoderBlock(D_MODEL, NUM_HEADS, D_FF, DROPOUT_RATE)
            for _ in range(NUM_LAYERS)
        ]
        self.final_layer = layers.Dense(VOCAB_SIZE)

    def call(self, x, training=False, mask=None):
        x = self.embedding(x)
        x = self.pos_encoding(x)
        for layer in self.dec_layers:
            x = layer(x, training=training, mask=mask)
        return self.final_layer(x)



app = FastAPI()
templates = Jinja2Templates(directory="templates")

def load_and_build(path):
    model = TransformerModel()
    model(tf.zeros((1, MAX_LEN))) 
    if os.path.exists(path):
        model.load_weights(path)
    return model


austen_model = load_and_build(AUSTEN_WEIGHTS)
sherlock_model = load_and_build(SHERLOCK_WEIGHTS)


try:
    with open(TOKENIZER_PATH, 'r', encoding='utf-8') as f:
        raw_content = json.load(f)
        
        if isinstance(raw_content, str):
            tokenizer = tokenizer_from_json(raw_content)
        else:
            
            tokenizer = tokenizer_from_json(json.dumps(raw_content))
    print("Tokenizer loaded successfully.")
except Exception as e:
    print(f"Error loading tokenizer: {e}")

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_db = FAISS.load_local(FAISS_INDEX_DIR, embeddings, allow_dangerous_deserialization=True)





def generate(model, prompt, length=20):
    seq = tokenizer.texts_to_sequences([prompt])[0]
    seq = tf.keras.preprocessing.sequence.pad_sequences([seq], maxlen=MAX_LEN, padding='post')
    
    for _ in range(length):
        preds = model.predict(seq, verbose=0)
        next_id = np.argmax(preds[:, -1, :], axis=-1)
        
        seq = np.append(seq[:, 1:], [next_id], axis=1).reshape(1, MAX_LEN)
        if next_id == 0: break
            
    return tokenizer.sequences_to_texts(seq)[0].replace("<OOV>", "").strip()

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/generate")
async def process(request: Request, prompt: str = Form(...)):
    
    out1 = generate(austen_model, prompt)
    
    
    out2 = generate(sherlock_model, prompt)
    
    
    docs = vector_db.similarity_search(prompt, k=2)
    context = " ".join([d.page_content for d in docs])
    out3 = generate(sherlock_model, f"{context} {prompt}")

    return templates.TemplateResponse("index.html", {
        "request": request, "prompt": prompt,
        "base": out1, "fine": out2, "rag": out3
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
