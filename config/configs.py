VOCAB_SIZE = 10000
MAX_LEN = 50  
BUFFER_SIZE = 20000

D_MODEL = 256
NUM_LAYERS = 4
NUM_HEADS = 8
D_FF = 1024
DROPOUT_RATE = 0.1

BATCH_SIZE = 64  
EPOCHS = 10
FINE_TUNING_EPOCHS=3
INITIAL_LEARNING_RATE = 5e-5 #1e-4
FINE_TUNING_LEARNING_RATE=1e-5

SAVE_PATH = "data/processed/decoder_only_weights.h5"
FINE_TUNED_PATH = "data/processed/transformer_sherlock.h5"

REPO_ID="Samkenneth4/austen-sherlock-slm"

WANDB_PROJECT = "austen-sherlock-slm"
WANDB_ENTITY = "awesome-mr2-na"