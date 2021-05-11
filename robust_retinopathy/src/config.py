DATA_DIR = "../../data/"
TRAIN_DIR = "../../data/train_images/"
CSV_PATH = "../../data/train.csv"
MODEL_PATH = "../../data/models/"


LEARNING_RATE = 1e-3
TRAIN_BATCH_SIZE = 4
VALID_BATCH_SIZE = 4
NUM_WORKERS = 4
EPOCHS = 2
IMG_WIDTH = 768
IMG_HEIGHT = 768

MODEL_NAME = "mobilenetv3_large_100"

MODEL_SAVE = MODEL_PATH + MODEL_NAME
USE_AMP = True
