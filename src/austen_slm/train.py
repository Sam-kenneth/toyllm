import sys
import os
import tensorflow as tf
import time
import json
import wandb
from wandb.integration.keras import WandbMetricsLogger

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import config.configs as configs
from src.austen_slm.data_loader import prepare_data
from src.austen_slm.model import create_model
from logger.logging import setup_logging
from exceptions.exception import PipelineException

logger = setup_logging(log_file_path='logs/train.log')

def setup_hardware():
                
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
            logger.info(f"Hardware setup complete: {len(gpus)} GPU(s) with Mixed Precision")
        except RuntimeError as e:
            logger.error(f"GPU Setup Error: {e}")
    else:
        logger.warning("No GPU detected. Running on CPU")

def main():

    
    os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'
    tf.config.optimizer.set_jit(False)
    try:
        logger.info("Starting training pipeline")
        
        
        wandb.init(
            project=configs.WANDB_PROJECT,
            config={
                "learning_rate": configs.INITIAL_LEARNING_RATE,
                "epochs": configs.EPOCHS,
                "batch_size": configs.BATCH_SIZE,
                "model_architecture": "Decoder-Only Transformer"
            }
        )
        
        
        setup_hardware()
        
        
        logger.info("Loading Austen corpus")
        X, y, tokenizer = prepare_data(
            vocab_size=configs.VOCAB_SIZE, 
            seq_length=configs.MAX_LEN
        )
        logger.info(f"Data prepared: {X.shape}")
        
        
        logger.info("Building Transformer model")
        model = create_model()
        
        
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=configs.INITIAL_LEARNING_RATE,
            clipnorm=1.0,
            beta_1=0.9,
            beta_2=0.98,
            epsilon=1e-9
        )
        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )
        early_stopping = EarlyStopping(
            monitor='val_loss', 
            patience=2, 
            verbose=1, 
            restore_best_weights=True
        )
        checkpoint = ModelCheckpoint(
            filepath=configs.SAVE_PATH,
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=True,
            mode='min',
            verbose=1
        )
        
        logger.info(f"Starting training: Batch={configs.BATCH_SIZE}, Epochs={configs.EPOCHS}")
        start_time = time.time()
        
        history = model.fit(
            X, y, 
            batch_size=configs.BATCH_SIZE, 
            epochs=configs.EPOCHS, 
            validation_split=0.1,
            verbose=1,
            callbacks=[WandbMetricsLogger(), early_stopping, checkpoint]
        )
        
        wandb.finish()
        
        train_time = time.time() - start_time
        logger.info(f"Training complete in {train_time:.2f}s. Final loss: {history.history['loss'][-1]:.4f}")
        
        
        os.makedirs(os.path.dirname(configs.SAVE_PATH), exist_ok=True)
        model.save_weights(configs.SAVE_PATH)
        logger.info(f"Weights saved to {configs.SAVE_PATH}")
        
        
        tokenizer_json = tokenizer.to_json()
        tokenizer_path = os.path.join('data', 'processed', 'tokenizer.json')
        with open(tokenizer_path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(tokenizer_json, ensure_ascii=False))
        logger.info(f"Tokenizer saved to {tokenizer_path}")
        
    except tf.errors.ResourceExhaustedError:
        logger.error("OOM Error: Reduce BATCH_SIZE in configs.py")
        wandb.finish()
        sys.exit(1)
    except Exception as e:
        wandb.finish()
        raise PipelineException("Training failed", sys)

if __name__ == "__main__":
    main()