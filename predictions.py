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

# # re-initialize the testing set generator, this time excluding the
# # `SimplePreprocessor`
# testGen = HDF5DatasetGenerator(config.TEST_HDF5, 64,
# 	preprocessors=[mp], classes=2)
# predictions = []
#
# # initialize the progress bar
# widgets = ["Evaluating: ", progressbar.Percentage(), " ",
# 	progressbar.Bar(), " ", progressbar.ETA()]
# pbar = progressbar.ProgressBar(maxval=testGen.numImages // 64,
# 	widgets=widgets).start()
#
# # loop over a single pass of the test data
# for (i, (images, labels)) in enumerate(testGen.generator(passes=1)):
# 	# loop over each of the individual images
# 	for image in images:
# 		# apply the crop preprocessor to the image to generate 10
# 		# separate crops, then convert them from images to arrays
# 		crops = cp.preprocess(image)
# 		crops = np.array([iap.preprocess(c) for c in crops],
# 			dtype="float32")
#
# 		# make predictions on the crops and then average them
# 		# together to obtain the final prediction
# 		pred = model.predict(crops)
# 		predictions.append(pred.mean(axis=0))
#
# 	# update the progress bar
# 	pbar.update(i)
#
# # compute the rank-1 accuracy
# pbar.finish()
# print("[INFO] predicting on test data (with crops)...")
# (rank1, _) = rank5_accuracy(predictions, testGen.db["labels"])
# print("[INFO] rank-1: {:.2f}%".format(rank1 * 100))
# testGen.close()
