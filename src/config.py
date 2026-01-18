import os
import torch

# data paths
DATA_DIR = os.path.join("/Users/keremoztopuz/Desktop/pneumonia_detector_project/data")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")
VAL_DIR = os.path.join(DATA_DIR, "val")

# model settings
MODEL_NAME = "convnext_t"
NUM_CLASSES = 2
DROP_RATE = 0.2

# training settings
LEARNING_RATE = 0.0001
BATCH_SIZE = 32
WEIGHT_DECAY = 0.05
EPOCHS = 50
PATIENCE = 10
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
CLASS_WEIGHTS = [2.0, 0.7]
SEED = 42

# augmentations
IMAGE_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# loss settings
FOCAL_GAMMA = 2

# saving paths
CHECKPOINT_DIR = "outputs/models"
MODEL_SAVE_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pth")