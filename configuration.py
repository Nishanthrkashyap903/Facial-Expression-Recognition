"""
Initialize all the configurations needed for the project.
"""
import os
 
# The path to the datasets. 
DATASET_DIR = f"datasets"
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
TEST_DIR = os.path.join(DATASET_DIR, "test")

# The split ratio for the training dataset. 
TRAIN_SIZE = 0.90
VAL_SIZE = 0.10
 
# Declare the batch size, the number of epochs and the learning rate. 
BATCH_SIZE = 16
NUM_EPOCHS = 50
LEARING_RATE = 1e-1