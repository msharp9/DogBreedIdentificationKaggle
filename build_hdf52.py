# USAGE
# python build_hdf52.py

# import the necessary packages
from config import config
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from sklearn.preprocessing import LabelEncoder
from pyimagesearch.io import HDF5DatasetWriter
from imutils import paths
import numpy as np
import pandas as pd
import progressbar
import random
import os

# Grab Label Data and build dictionary/grab class names
data = pd.read_csv('labels.csv')
ids_data = data['id']
labels_data = data['breed']
dict_data = data.set_index('id')['breed'].to_dict()
classNames = [str(x) for x in np.unique(labels_data)]

# grab the list of images that we'll be describing then randomly
# shuffle them to allow for easy training and testing splits via
# array slicing during training time
print("[INFO] loading images...")
imagePaths = list(paths.list_images(config.IMAGES_PATH))
random.shuffle(imagePaths) # pre-shuffled, nothing wrong with shuffling again
ids = [os.path.splitext(os.path.basename(path))[0] for path in imagePaths]
labels = [dict_data[i] for i in ids]

# encode the labels
le = LabelEncoder()
labels = le.fit_transform(labels)

# initialize the HDF5 dataset writer, then store the class label names in the dataset
dataset = HDF5DatasetWriter((len(imagePaths), config.INPUT_SIZE, config.INPUT_SIZE, 3),
	config.TRAIN_HDF5)
dataset.storeClassLabels(le.classes_)

# initialize the progress bar
widgets = ["Saving Images: ", progressbar.Percentage(), " ",
	progressbar.Bar(), " ", progressbar.ETA()]
pbar = progressbar.ProgressBar(maxval=len(imagePaths),
	widgets=widgets).start()

# loop over the images in batches
for i in np.arange(0, len(imagePaths)):
    # Grab values
	imagePath = imagePaths[i]
	label = labels[i]
	_id = ids[i]

	# load the input image using the Keras helper utility
	# while ensuring the image is resized
	image = load_img(imagePath, target_size=(config.INPUT_SIZE, config.INPUT_SIZE))
	image = img_to_array(image)

	# add the features and labels to our HDF5 dataset
	dataset.add([image], [label], [_id])
	pbar.update(i)

# close the dataset
dataset.close()
pbar.finish()
