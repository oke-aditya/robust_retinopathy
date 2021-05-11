DATA_DIR = "../../data/"
TRAIN_DIR = "../../data/train_images/"
CSV_PATH = "../../data/train.csv"
MODEL_PATH = "../../data/models/"

TRAIN_SPLIT = 0.8
# Automatically
# VAL_SPLIT = 0.2

# Over 1 GPU
LEARNING_RATE = 1e-4
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 32

# 1 Processor x 2 Cores
NUM_WORKERS = 2
EPOCHS = 10

IMG_WIDTH = 768
IMG_HEIGHT = 768

MODEL_NAME = "mobilenetv3_large_100"

MODEL_SAVE = MODEL_PATH + MODEL_NAME

# Use Mixed Precision Training
USE_AMP = True
