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
# i = int(db["labels"].shape[0] * 0.75)

sp = SimplePreprocessor(299, 299)
iap = ImageToArrayPreprocessor()
# initialize the training and validation dataset generators
trainGen = HDF5DatasetGenerator(config.TRAIN_HDF5, 32, aug=aug,
	preprocessors=[sp, iap], classes=config.NUM_CLASSES, set="train")
valGen = HDF5DatasetGenerator(config.TRAIN_HDF5, 32, aug=aug2,
	preprocessors=[sp, iap], classes=config.NUM_CLASSES, set="val")

# image = np.expand_dims(image, axis=0)
# image = imagenet_utils.preprocess_input(image)
# x = xception.preprocess_input(np.expand_dims(img.copy(), axis=0))
# POOLING = 'avg'
# xception_bottleneck = xception.Xception(weights='imagenet', include_top=False, pooling=POOLING)

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


# # define the set of parameters that we want to tune then start a
# # grid search where we evaluate our model for each value of C
# print("[INFO] tuning hyperparameters...")
# params = {"C": [0.0001, 0.001, 0.01, 0.1, 1.0]}
# model = GridSearchCV(LogisticRegression(), params, n_jobs=-1)
# model.fit(db["features"][:i], db["labels"][:i])
# print("[INFO] best hyperparameters: {}".format(model.best_params_))
#
# # generate a classification report for the model
# print("[INFO] evaluating...")
# preds = model.predict(db["features"][i:])
# print(classification_report(db["labels"][i:], preds,
# 	target_names=db["label_names"]))
#
# # compute the raw accuracy with extra precision
# acc = accuracy_score(db["labels"][i:], preds)
# print("[INFO] score: {}".format(acc))
#
# # serialize the model to disk
# print("[INFO] saving model...")
# f = open(config.MODEL_PATH, "wb")
# f.write(pickle.dumps(model.best_estimator_))
# f.close()
#
# # close the database
# db.close()
