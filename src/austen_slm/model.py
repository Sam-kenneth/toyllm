import tensorflow as tf
from tensorflow.keras import layers
import config.configs as configs
import numpy as np

def create_look_ahead_mask(size):
    
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask

class PositionalEncoding(layers.Layer):
    def __init__(self, max_len, d_model):
        super().__init__()
        import numpy as np
        
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
        
        pe = tf.cast(self.pe[:, :seq_len, :], dtype=x.dtype)
        return x + pe

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



class Transformer(tf.keras.Model):
    def __init__(self): 
        super().__init__()
        self.embedding = layers.Embedding(configs.VOCAB_SIZE, configs.D_MODEL)
        self.pos_encoding = PositionalEncoding(configs.MAX_LEN, configs.D_MODEL)
        
        self.dec_layers = [
            DecoderBlock(configs.D_MODEL, configs.NUM_HEADS, configs.D_FF, configs.DROPOUT_RATE)
            for _ in range(configs.NUM_LAYERS)
        ]
        
        self.final_layer = layers.Dense(configs.VOCAB_SIZE, dtype='float32')

    def call(self, x, training=False):
        mask = create_look_ahead_mask(tf.shape(x)[1])
        x = self.embedding(x)
        x = self.pos_encoding(x)
        for layer in self.dec_layers:
            x = layer(x, mask=mask, training=training)
        return self.final_layer(x)

def create_model():
    return Transformer()