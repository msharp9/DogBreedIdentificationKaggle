# USAGE
# python predictions.py

# import the necessary packages
from config import config
from pyimagesearch.preprocessing import ImageToArrayPreprocessor
from pyimagesearch.preprocessing import SimplePreprocessor
from pyimagesearch.preprocessing import MeanPreprocessor
from pyimagesearch.preprocessing import CropPreprocessor
from pyimagesearch.io import HDF5DatasetGenerator
from pyimagesearch.utils.ranked import rank5_accuracy
from keras.applications import xception
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import progressbar
import json


# initialize the image preprocessors
sp = SimplePreprocessor(299, 299)
iap = ImageToArrayPreprocessor()
aug = ImageDataGenerator(preprocessing_function=xception.preprocess_input)

# load the pretrained network
print("[INFO] loading model...")
model = load_model(config.MODEL_PATH)

# initialize the testing dataset generator, then make predictions on
# the testing data
print("[INFO] predicting on test data (no crops)...")
testGen = HDF5DatasetGenerator(config.TRAIN_HDF5, 64, aug=aug,
	preprocessors=[sp, iap], classes=config.NUM_CLASSES)
predictions = model.predict_generator(testGen.generator(),
	steps=testGen.numImages // 64)

# compute the rank-1 and rank-5 accuracies
(rank1, rank5) = rank5_accuracy(predictions, testGen.db["labels"])
print(predictions)
print("[INFO] rank-1: {:.2f}%".format(rank1 * 100))
print("[INFO] rank-5: {:.2f}%".format(rank5 * 100))
testGen.close()
