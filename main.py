# Dependencies
from utils import *
from config import *
import tensorflow as tf
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Loading and preprocessing train images

lr_ds = image_dataset_from_directory(
    directory=TRAIN_LR_DIR,
    batch_size=BATCH_SIZE,
    labels=None,
    image_size=IMAGE_SIZE)

hr_ds = image_dataset_from_directory(
    directory=TRAIN_HR_DIR,
    batch_size=BATCH_SIZE,
    labels=None,
    image_size=IMAGE_SIZE)

normalised_lr = lr_ds.map(NORMALIZATION_FUNC)
normalised_hr = hr_ds.map(NORMALIZATION_FUNC)

train_ds = tf.data.Dataset.zip((normalised_lr,normalised_hr))


# Loading and preprocessing val images

lr_ds = image_dataset_from_directory(
    directory=VAL_LR_DIR,
    batch_size=BATCH_SIZE,
    labels=None,
    image_size=IMAGE_SIZE)

hr_ds = image_dataset_from_directory(
    directory=VAL_HR_DIR,
    batch_size=BATCH_SIZE,
    labels=None,
    image_size=IMAGE_SIZE)

normalised_lr = lr_ds.map(NORMALIZATION_FUNC)
normalised_hr = hr_ds.map(NORMALIZATION_FUNC)

val_ds = tf.data.Dataset.zip((normalised_lr,normalised_hr))