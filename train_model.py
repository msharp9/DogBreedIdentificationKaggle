# USAGE
# python train_model.py --db ../datasets/kaggle_dogs_vs_cats/hdf5/features.hdf5 --model dogs_vs_cats.pickle

import matplotlib
matplotlib.use("Agg")

# import the necessary packages
from config import config
from pyimagesearch.callbacks import TrainingMonitor
from pyimagesearch.io import HDF5DatasetGenerator
from pyimagesearch.preprocessing import ImageToArrayPreprocessor
from keras.applications import xception
from keras.applications import imagenet_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
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

# open the HDF5 database for reading then determine the index of
# the training and testing split, provided that this data was
# already shuffled *prior* to writing it to disk
db = h5py.File(config.TRAIN_HDF5, "r")
i = int(db["labels"].shape[0] * 0.75)

iap = ImageToArrayPreprocessor()
# initialize the training and validation dataset generators
trainGen = HDF5DatasetGenerator(config.TRAIN_HDF5, 64, aug=aug,
	preprocessors=[pp, mp, iap], classes=2)
valGen = HDF5DatasetGenerator(config.VAL_HDF5, 64,
	preprocessors=[sp, mp, iap], classes=2)

image = np.expand_dims(image, axis=0)
image = imagenet_utils.preprocess_input(image)
x = xception.preprocess_input(np.expand_dims(img.copy(), axis=0))
POOLING = 'avg'
xception_bottleneck = xception.Xception(weights='imagenet', include_top=False, pooling=POOLING)

# construct the set of callbacks
path = os.path.sep.join([config.OUTPUT_PATH, "{}.png".format(
	os.getpid())])
callbacks = [TrainingMonitor(path)]

# train the network
model.fit_generator(
	trainGen.generator(),
	steps_per_epoch=trainGen.numImages // 64,
	validation_data=valGen.generator(),
	validation_steps=valGen.numImages // 64,
	epochs=50,
	max_q_size=64 * 2,
	callbacks=callbacks, verbose=1)

# save the model to file
print("[INFO] serializing model...")
model.save(config.MODEL_PATH, overwrite=True)

# close the HDF5 datasets
trainGen.close()
valGen.close()

# define the set of parameters that we want to tune then start a
# grid search where we evaluate our model for each value of C
print("[INFO] tuning hyperparameters...")
params = {"C": [0.0001, 0.001, 0.01, 0.1, 1.0]}
model = GridSearchCV(LogisticRegression(), params, cv=3,
	n_jobs=-1)
model.fit(db["features"][:i], db["labels"][:i])
print("[INFO] best hyperparameters: {}".format(model.best_params_))

# generate a classification report for the model
print("[INFO] evaluating...")
preds = model.predict(db["features"][i:])
print(classification_report(db["labels"][i:], preds,
	target_names=db["label_names"]))

# compute the raw accuracy with extra precision
acc = accuracy_score(db["labels"][i:], preds)
print("[INFO] score: {}".format(acc))

# serialize the model to disk
print("[INFO] saving model...")
f = open(config.MODEL_PATH, "wb")
f.write(pickle.dumps(model.best_estimator_))
f.close()

# close the database
db.close()
