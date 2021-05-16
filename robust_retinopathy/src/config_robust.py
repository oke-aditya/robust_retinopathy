DATA_DIR = "../../data/"
TRAIN_DIR = "../../data/train_images/"
CSV_PATH = "../../data/train.csv"
MODEL_PATH = "../../data/models/"

TRAIN_SPLIT = 0.8
# Automaticllay
# VAL_SPLIT = 0.2

LEARNING_RATE = 1e-4
TRAIN_BATCH_SIZE = 32
# TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 32
# VALID_BATCH_SIZE = 16

NUM_WORKERS = 2
EPOCHS = 15

IMG_WIDTH = 224
IMG_HEIGHT = 224

# MODEL_NAME = "vit_base_patch16_224"
MODEL_NAME = "vit_deit_base_patch16_224"

MODEL_SAVE = MODEL_PATH + MODEL_NAME

# Use Mixed Precision Training
USE_AMP = True
