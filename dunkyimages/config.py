################################################################
# Written  by Opiyo Geoffrey Duncan: Deep Learning             #
################################################################
# importing the necessary packages from the library
import os

# initializing the path to the input directory containing our dataset
# of images
DATASET_PATH = "Cyclone_Database"

# initializing the class labels in the dataset
CLASSES = ["Cyclone", "Earthquake", "Flood", "Wildfire"]

# defining the size of the training and validation set, this comes from the
# train split and testing splits respectively.
TRAIN_SPLIT = 0.75
VAL_SPLIT = 0.1
TEST_SPLIT = 0.25

# defining the minimum learning rate, maximum learning rate, batch size,
# step size, CLR method, and number of epochs
MIN_LR = 1e-6
MAX_LR = 1e-4
BATCH_SIZE = 32
STEP_SIZE = 8
CLR_METHOD = "triangular"
NUM_EPOCHS = 48

# setting the path to the serialized model after training
MODEL_PATH = os.path.sep.join(["output", "natural_disasters.model"])

# defining the path to the output learning rate finder plot, training
# history plot and cyclical learning rate plot
LRFIND_PLOT_PATH = os.path.sep.join(["output", "lrfind_plot.png"])
TRAINING_PLOT_PATH = os.path.sep.join(["output", "training_plot.png"])
CLR_PLOT_PATH = os.path.sep.join(["output", "clr_plot.png"])