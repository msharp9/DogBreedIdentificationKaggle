# USAGE
# python submission.py

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
import pandas as pd
import progressbar
import json
import h5py

# initialize the image preprocessors
sp = SimplePreprocessor(299, 299)
iap = ImageToArrayPreprocessor()
aug = ImageDataGenerator(preprocessing_function=xception.preprocess_input)

# # preprocess images
# procImages = []
# # loop over the images
# for image in images:
# 	image = sp.preprocess(image)
# 	image = iap.preprocess(image)
# 	image = xception.preprocess_input(image)
# 	procImages.append(image)
#
# # update the images array to be the processed images
# images = np.array(procImages)

# load the pretrained network
print("[INFO] loading model...")
model = load_model(config.MODEL_PATH)

# initialize the testing dataset generator, then make predictions on
# the testing data
print("[INFO] predicting on test data...")
testGen = HDF5DatasetGenerator(config.TEST_HDF5, 64, aug=aug,
	preprocessors=[sp, iap], classes=config.NUM_CLASSES)
# print(testGen.numImages, testGen.numImages//64)
predictions = model.predict_generator(testGen.generator(),
	steps=162)
testGen.close()

# Create Pandas dataframe and save to csv
db = h5py.File(config.TEST_HDF5)
cols = np.append('id', db["label_names"][:])
ids = np.asarray(db["ids"])
print(db["ids"].shape, db["images"].shape)
print(ids.shape, predictions.shape)
results = np.hstack((ids[:, np.newaxis], predictions))
df = pd.DataFrame(data=results, columns=cols)
df.to_csv('output/submission.csv', index=False)
