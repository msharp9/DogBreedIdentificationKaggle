# USAGE
# python finetune_flowers17.py --dataset ../datasets/flowers17/images \
# 	--model flowers17.model

# import the necessary packages
from config import config
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from pyimagesearch.preprocessing import ImageToArrayPreprocessor
from pyimagesearch.preprocessing import AspectAwarePreprocessor
from pyimagesearch.datasets import SimpleDatasetLoader
from pyimagesearch.nn.conv import FCHeadNet
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import RMSprop
from keras.optimizers import SGD
from keras.applications import VGG16
from keras.applications import xception
from keras.layers import Input
from keras.models import Model
from imutils import paths
import numpy as np
import pandas as pd
import argparse
import os

# construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
	height_shift_range=0.1, shear_range=0.1, zoom_range=0.2,
	horizontal_flip=True, fill_mode="nearest")

# grab the list of images that we'll be describing, then extract
# the class label names from the image paths
print("[INFO] loading images...")
imagePaths = list(paths.list_images(config.IMAGES_PATH))
data = pd.read_csv('labels.csv')
# ids = data['id']
classNames = data['breed']
classNames = [str(x) for x in np.unique(classNames)]
data_dict = data.set_index('id')['breed'].to_dict()

# le = LabelEncoder()
# labels = le.fit_transform(labels)
# labels_inv = le.inverse_transform(labels)
# classNames = [pt.split(os.path.sep)[-2] for pt in imagePaths]
# classNames = [str(x) for x in np.unique(classNames)]

# initialize the image preprocessors
aap = AspectAwarePreprocessor(config.INPUT_SIZE, config.INPUT_SIZE)
iap = ImageToArrayPreprocessor()

# load the dataset from disk then scale the raw pixel intensities to
# the range [0, 1]
sdl = SimpleDatasetLoader(preprocessors=[aap, iap])
(data, ids) = sdl.load(imagePaths, verbose=500)
labels = [data_dict[i] for i in ids]
data = data.astype("float") / 255.0

# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=0.25, stratify=labels)

# convert the labels to vectors
trainY = LabelBinarizer().fit_transform(trainY)
testY = LabelBinarizer().fit_transform(testY)

# load the VGG16 network, ensuring the head FC layer sets are left
# off
baseModel = xception.Xception(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(config.INPUT_SIZE, config.INPUT_SIZE, 3)))

# initialize the new head of the network, a set of FC layers
# followed by a softmax classifier
headModel = FCHeadNet.build(baseModel, config.NUM_CLASSES, 256)

# place the head FC model on top of the base model -- this will
# become the actual model we will train
model = Model(inputs=baseModel.input, outputs=headModel)

# loop over all layers in the base model and freeze them so they
# will *not* be updated during the training process
for layer in baseModel.layers:
	layer.trainable = False

# compile our model (this needs to be done after our setting our
# layers to being non-trainable
print("[INFO] compiling model...")
opt = RMSprop(lr=0.001)
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# train the head of the network for a few epochs (all other
# layers are frozen) -- this will allow the new FC layers to
# start to become initialized with actual "learned" values
# versus pure random
print("[INFO] training head...")
model.fit_generator(aug.flow(trainX, trainY, batch_size=16),
	validation_data=(testX, testY), epochs=25,
	steps_per_epoch=len(trainX) // 16, verbose=1)

# evaluate the network after initialization
print("[INFO] evaluating after initialization...")
predictions = model.predict(testX, batch_size=16)
print(classification_report(testY.argmax(axis=1),
	predictions.argmax(axis=1), target_names=classNames))

# # now that the head FC layers have been trained/initialized, lets
# # unfreeze the final set of CONV layers and make them trainable
# for layer in baseModel.layers[15:]:
# 	layer.trainable = True
#
# # for the changes to the model to take affect we need to recompile
# # the model, this time using SGD with a *very* small learning rate
# print("[INFO] re-compiling model...")
# opt = SGD(lr=0.001)
# model.compile(loss="categorical_crossentropy", optimizer=opt,
# 	metrics=["accuracy"])
#
# # train the model again, this time fine-tuning *both* the final set
# # of CONV layers along with our set of FC layers
# print("[INFO] fine-tuning model...")
# model.fit_generator(aug.flow(trainX, trainY, batch_size=32),
# 	validation_data=(testX, testY), epochs=100,
# 	steps_per_epoch=len(trainX) // 32, verbose=1)
#
# # evaluate the network on the fine-tuned model
# print("[INFO] evaluating after fine-tuning...")
# predictions = model.predict(testX, batch_size=32)
# print(classification_report(testY.argmax(axis=1),
# 	predictions.argmax(axis=1), target_names=classNames))

# save the model to disk
print("[INFO] serializing model...")
model.save(config.MODEL_PATH)
