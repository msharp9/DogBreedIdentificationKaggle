# USAGE
# python build_hdf5.py --image_path .. --output output

# import the necessary packages
from config import config
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from pyimagesearch.preprocessing import AspectAwarePreprocessor
from pyimagesearch.io import HDF5DatasetWriter
from imutils import paths
import numpy as np
import pandas as pd
import progressbar
import argparse
import json
import cv2
import os

# # construct the argument parser, only need an image input
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image_path", required=True, help="path to the image files")
# ap.add_argument("-o", "--output", required=True, help="path for outputs")
# ap.add_argument("-s", "--buffer-size", type=int, default=1000, help="size of feature extraction buffer")
# args = vars(ap.parse_args())

# Grab Label Data and build dictionary/grab class names
data = pd.read_csv('labels.csv')
ids_data = data['id']
labels_data = data['breed']
dict_data = data.set_index('id')['breed'].to_dict()
classNames = [str(x) for x in np.unique(labels_data)]

# grab the paths to the images
imagePaths = list(paths.list_images(config.IMAGES_PATH))
ids = [os.path.splitext(os.path.basename(path))[0] for path in imagePaths]
labels = [dict_data[i] for i in ids]

# Kaggle is nice enough to prepare data for you (it's also already randomized)
# print(ids_data == ids) # True
# print(labels_data == labels) # True

# Encode Labels
le = LabelEncoder()
labels = le.fit_transform(labels)

# perform stratified sampling from the training set
split = train_test_split(images, labels, ids,
	test_size=round(len(images)*0.15), stratify=labels)
(trainPaths, testPaths, trainLabels, testLabels, trainIds, testIds) = split

# perform another stratified sampling, this time to build the validation data
split = train_test_split(trainPaths, trainLabels, trainIds,
	test_size=round(len(imagePaths)*0.15), stratify=trainLabels)
(trainPaths, valPaths, trainLabels, valLabels, trainIds, valIds) = split

# construct a list pairing the training, validation, and testing
# image paths along with their corresponding labels and output HDF5 files
datasets = [
	("train", trainPaths, trainLabels, trainIds, config.TRAIN_HDF5),
	("val", valPaths, valLabels, valIds, config.VAL_HDF5),
	("test", testPaths, testLabels, testIds, config.TEST_HDF5)]

# initialize the image pre-processor and the lists of RGB channel averages
aap = AspectAwarePreprocessor(config.INPUT_SIZE, config.INPUT_SIZE)
(R, G, B) = ([], [], [])

# loop over the dataset tuples
for (dType, paths, labels, ids, outputPath) in datasets:
	# create HDF5 writer
	print("[INFO] building {}...".format(outputPath))
	writer = HDF5DatasetWriter((len(paths), config.INPUT_SIZE, config.INPUT_SIZE, 3),
		outputPath, bufSize=args["buffer_size"])
	writer.storeClassLabels(le.classes_)

	# initialize the progress bar
	widgets = ["Building Dataset: ", progressbar.Percentage(), " ",
		progressbar.Bar(), " ", progressbar.ETA()]
	pbar = progressbar.ProgressBar(maxval=len(paths),
		widgets=widgets).start()

	# loop over the image paths
	for (i, (path, label, _id)) in enumerate(zip(paths, labels, ids)):
		# load the image and process it
		image = cv2.imread(path)
		image = aap.preprocess(image)

		# if we are building the training dataset, then compute the
		# mean of each channel in the image, then update the respective lists
		if dType == "train":
			(b, g, r) = cv2.mean(image)[:3]
			R.append(r)
			G.append(g)
			B.append(b)

		# add the image and label # to the HDF5 dataset
		writer.add([image], [label], [_id])
		pbar.update(i)

	# close the HDF5 writer
	pbar.finish()
	writer.close()

# construct a dictionary of averages, then serialize the means to a JSON file
print("[INFO] serializing means...")
D = {"R": np.mean(R), "G": np.mean(G), "B": np.mean(B)}

f = open(config.DATASET_MEAN, "w")
f.write(json.dumps(D))
f.close()
