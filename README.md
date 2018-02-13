Kaggle Challenge for Dog Breed Identification.  This is a currently a playground competition. 120 different breeds and ~10K images.  
It's about 60-90 images a dog breed, which is a low training set, going to have to use some type of augmentation.
Going to focus on using Xception/InceptionV3 models as they proven to be most effect on a sample in the top voted kernal.


First attempted just training it straight but loading 10K images is too much to do in memory for my RAM, will need to take advantage of hdf5. Also, since I'm joining this challenge late I won't dive deep into fine tuning the models, but do simple top level transfer learning.  This should help w/ time, especially since I don't have a GPU.

Trained a very simple model using xception + a simple head on top.  Just want to see where it gets me.  Probably don't have time to mess around with it more though.

It was a low score but surprisingly not last.  Since it's only a playground challenge and it ends soon won't fully train model. (You know, more than 1 epoch :D)

DogBreedTrainer.py was just a first pass and a good place to start if you have a beefy computer.
build_hdf5.py scripts to build the different databases
train_model.py to create the model
predictions.py to evaluate the model
submission.py to prepare the csv file in the correct format for the Kaggle submission
