"""
Initialize all the configurations needed for the project.
"""
import os
 
# The path to the datasets. 
DATASET_DIR = f"/home/stu9/s10/nk4349/csci-631/project/datasets"
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
TEST_DIR = os.path.join(DATASET_DIR, "test")

# The split ratio for the training dataset. 
TRAIN_SIZE = 0.90
VAL_SIZE = 0.10
 
# Declare the batch size, the number of epochs and the learning rate. 
BATCH_SIZE = 32
NUM_EPOCHS = 50
WEIGHT_DECAY = 0.0001