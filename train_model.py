# import the necessary packages
import argparse
import os
from pickle import dump

import matplotlib.pyplot as plt
import numpy as np
import cv2
from imutils import paths
from keras.optimizers import SGD
from keras.preprocessing.image import img_to_array
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

from nn.lenet import LeNet
# Set the random seed
from utils.captcha_helper import preprocess

np.random.seed(0)

# Define input arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="path to input dataset")
ap.add_argument("-m", "--model", required=True,
                help="path to output model")
args = vars(ap.parse_args())

# initialize the data and labels
data = []
labels = []

# loop over the input images
for image_path in paths.list_images(args["dataset"]):
    # load the image, pre-processing it, and add it to the data list
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = preprocess(image, 28, 28)
    image = img_to_array(image)
    data.append(image)

    # extract the class label from the image path and update the 
    # labels list
    label = image_path.split(os.path.sep)[-2]
    labels.append(label)

# scale the pixels to the range [0, 1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# partition the data into training and testing splits using 75% of 
# the data for training and remaining 25% for testing
(train_X, test_X, train_Y, test_Y) = train_test_split(data,
                                                      labels, test_size=0.25,
                                                      random_state=42)

# convert the labels from integers to vectors
lb = LabelBinarizer().fit(train_Y)
train_Y = lb.transform(train_Y)
test_Y = lb.transform(test_Y)

# initialize the model 
print("[INFO] compiling model...")
num_classes = np.unique(labels).shape[0]
model = LeNet.build(width=28, height=28, depth=1, classes=num_classes)
num_epochs = 20
opt = SGD(lr=0.01, momentum=0.9)
model.compile(loss="categorical_crossentropy", optimizer=opt,
              metrics=["accuracy"])

# train the network
print("[INFO] training network...")
batch_size = 32
H = model.fit(train_X, train_Y, validation_data=(test_X, test_Y),
              batch_size=batch_size, epochs=num_epochs, verbose=1)

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(test_X, batch_size=batch_size)
print(classification_report(test_Y.argmax(axis=1),
                            predictions.argmax(axis=1),
                            target_names=lb.classes_))

# save the model and label binarizer to disk
print("[INFO] serializing network...")
model.save(args["model"])
dump(lb, open(os.path.join(args["model"], "label_binarizer.pkl"), 'wb'))

# plot the training + testing loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, num_epochs), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, num_epochs), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, num_epochs), H.history["accuracy"], label="accuracy")
plt.plot(np.arange(0, num_epochs), H.history["val_accuracy"],
         label="val_accuracy")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(os.path.join(args['model'], "result.png"))
plt.show()
