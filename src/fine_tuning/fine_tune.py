import sys
import os
import tensorflow as tf
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)
import config.configs as configs
from src.austen_slm.data_loader import prepare_data
from src.austen_slm.model import create_model
from logger.logging import setup_logging
from exceptions.exception import PipelineException
import wandb
from wandb.integration.keras import WandbMetricsLogger


logger = setup_logging(log_file_path='logs/fine_tune.log')
wandb.init(
    project=configs.WANDB_PROJECT,
    config={
        "learning_rate": configs.FINE_TUNING_LEARNING_RATE,
        "epochs": configs.FINE_TUNING_EPOCHS,
        "batch_size": configs.BATCH_SIZE,
        "model_architecture": "Decoder-Only Transformer",
        "corpus": "Sherlock Holmes",
        "fine_tuning": True

    }
)

def main():
    try:
        logger.info("="*50)
        logger.info("Starting Fine-Tuning Pipeline")
        logger.info("="*50)
        
        
        logger.info("Step 1: Configuring GPU and Mixed Precision")
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
            logger.info(f"GPU configured: {len(gpus)} GPU(s) detected")
            logger.info("Mixed precision (float16) enabled")
        else:
            logger.warning("No GPU detected. Training will run on CPU")

        
        logger.info("Step 2: Loading Sherlock Holmes corpus")
        try:
            X, y, _ = prepare_data(configs.VOCAB_SIZE, configs.MAX_LEN, corpus_type='sherlock')
            logger.info(f"Data loaded successfully - Shape: X={X.shape}, y={y.shape}")
        except Exception as e:
            raise PipelineException("Failed to load Sherlock Holmes data", sys)

        
        logger.info("Step 3: Initializing model and loading base weights")
        try:
            model = create_model()
            
            
            model(tf.zeros((1, configs.MAX_LEN)))
            logger.info("Model architecture created")
            
            if os.path.exists(configs.SAVE_PATH):
                model.load_weights(configs.SAVE_PATH)
                logger.info(f"Loaded base Austen weights from {configs.SAVE_PATH}")
            else:
                logger.warning(f"Base weights not found at {configs.SAVE_PATH}")
                logger.warning("Fine-tuning will start from scratch")
        except Exception as e:
            raise PipelineException("Failed to initialize model or load weights", sys)

        
        logger.info("Step 4: Compiling model with fine-tuning learning rate")
        try:
            optimizer = tf.keras.optimizers.Adam(learning_rate=configs.FINE_TUNING_LEARNING_RATE)
            model.compile(
                optimizer=optimizer, 
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy']
            )
            logger.info("Model compiled with learning rate: 1e-5")
        except Exception as e:
            raise PipelineException("Failed to compile model", sys)

        
        logger.info("Step 5: Starting fine-tuning on Sherlock Holmes corpus")
        logger.info(f"Training configuration: Batch size={configs.BATCH_SIZE}, Epochs=3")
        try:
            history = model.fit(
                X, y, 
                batch_size=configs.BATCH_SIZE, 
                epochs=3,
                verbose=1,
                callbacks=[WandbMetricsLogger()]
            )
            wandb.finish()
            logger.info("Fine-tuning completed successfully")
            logger.info(f"Final loss: {history.history['loss'][-1]:.4f}")
        except Exception as e:
            raise PipelineException("Fine-tuning training failed", sys)

        
        logger.info("Step 6: Saving fine-tuned model")
        try:
            FINE_TUNED_PATH = configs.FINE_TUNED_PATH
            
            
            os.makedirs(os.path.dirname(FINE_TUNED_PATH), exist_ok=True)
            
            model.save_weights(FINE_TUNED_PATH)
            logger.info(f"Fine-tuned model saved to {FINE_TUNED_PATH}")
        except Exception as e:
            raise PipelineException("Failed to save fine-tuned model", sys)

        logger.info("="*50)
        logger.info("Fine-Tuning Pipeline Completed Successfully!")
        logger.info("="*50)
        
    except PipelineException as pe:
        logger.error(f"Pipeline Exception: {pe}")
        logger.error("Fine-tuning pipeline failed. Check logs for details.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise PipelineException("Unexpected error in fine-tuning pipeline", sys)

if __name__ == "__main__":
    main()