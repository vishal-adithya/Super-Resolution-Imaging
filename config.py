
#----------------------#----------------------#-----------------{ DIR PATHS }-----------------#----------------------#

TRAIN_LR_DIR = "Data/Train/DIV2K_train_LR_bicubic_X4"
TRAIN_HR_DIR = "Data/Train/DIV2K_train_HR"
VAL_LR_DIR = "Data/Validation/DIV2K_valid_LR_bicubic_X4"
VAL_HR_DIR = "Data/Validation/DIV2K_valid_HR"

MODEL_CHECKPOINT_CALLBACK = "Checkpoint"

#----------------------#----------------------#-----------------{ IMAGE CONFIG }-----------------#----------------------#

IMAGE_SIZE = (256,256)
IMAGE_SIZE_CHL = (256,256,3)

#----------------------#----------------------#-----------------{ MODEL PARAMETERS }-----------------#----------------------#

BATCH_SIZE = 32
VAL_BATCH_SIZE = 16
OPTIMIZER = "adam"
METRICS = ["accuracy"]
EPOCHS = 50
LOSS = "mse"