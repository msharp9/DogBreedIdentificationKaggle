# USAGE
# python train_model.py

import matplotlib
matplotlib.use("Agg")

# import the necessary packages
from config import config
from pyimagesearch.callbacks import TrainingMonitor
from pyimagesearch.io import HDF5DatasetGenerator
from pyimagesearch.preprocessing import ImageToArrayPreprocessor
from pyimagesearch.preprocessing import SimplePreprocessor
from pyimagesearch.nn.conv import FCHeadNet
from keras.applications import xception
from keras.applications import imagenet_utils
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.optimizers import RMSprop
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import numpy as np
import pickle
import h5py
import os


# construct the training image generator for data augmentation
aug = ImageDataGenerator(preprocessing_function=xception.preprocess_input,
	rotation_range=30, zoom_range=0.2,
	width_shift_range=0.2, height_shift_range=0.2, shear_range=0.1,
	horizontal_flip=True, fill_mode="nearest")
aug2 = ImageDataGenerator(preprocessing_function=xception.preprocess_input)

# # open the HDF5 database for reading then determine the index of
# # the training and testing split, provided that this data was
# # already shuffled *prior* to writing it to disk
# db = h5py.File(config.TRAIN_HDF5, "r")
# print(db["label_names"], len(db["label_names"]))

sp = SimplePreprocessor(299, 299)
iap = ImageToArrayPreprocessor()
# initialize the training and validation dataset generators
trainGen = HDF5DatasetGenerator(config.TRAIN_HDF5, 32, aug=aug,
	preprocessors=[sp, iap], classes=config.NUM_CLASSES, set="train")
valGen = HDF5DatasetGenerator(config.TRAIN_HDF5, 32, aug=aug2,
	preprocessors=[sp, iap], classes=config.NUM_CLASSES, set="val")

# construct the set of callbacks
path = os.path.sep.join([config.OUTPUT_PATH, "{}.png".format(
	os.getpid())])
callbacks = [TrainingMonitor(path)]

# build model
xception_model = xception.Xception(input_shape=(299,299,3), weights='imagenet', include_top=False)#, pooling='avg')
headModel = FCHeadNet.build(xception_model, config.NUM_CLASSES, 1024)
model = Model(inputs=xception_model.input, outputs=headModel)
# freeze the xception model layers
for layer in xception_model.layers:
	layer.trainable = False

# compile model
opt = RMSprop(lr=0.001)
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# train the network
model.fit_generator(
	trainGen.generator(),
	steps_per_epoch=trainGen.numImages // 32,
	validation_data=valGen.generator(),
	validation_steps=(valGen.numImages-valGen.startImages) // 32,
	epochs=1,
	callbacks=callbacks, verbose=1)

# save the model to file
print("[INFO] serializing model...")
model.save(config.MODEL_PATH, overwrite=True)

# close the HDF5 datasets
trainGen.close()
valGen.close()
