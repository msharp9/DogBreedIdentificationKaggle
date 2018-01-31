# import the necessary packages
from keras.applications import VGG16
from keras.applications import ResNet50
from keras.applications import xception
from keras.applications import inception_v3
import argparse

# load the VGG16 network
print("[INFO] loading network...")
modelVGG16 = VGG16(weights="imagenet")
modelResNet50 = ResNet50(weights="imagenet")
modelx = xception.Xception(weights="imagenet")
modeli = inception_v3.InceptionV3(weights="imagenet")
print("[INFO] showing layers...")

models = [modelVGG16, modelResNet50, modelx, modeli]

# loop over the layers in the network and display them to the console
for model in models:
	for (i, layer) in enumerate(model.layers):
		print("[INFO] {}\t{}".format(i, layer))
