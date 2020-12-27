# import the necessary packages
import argparse
import os
from pickle import load

import cv2
import numpy as np
from imutils import paths
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from utils.captcha_helper import extract_the_characters
from utils.captcha_helper import preprocess


def run():
    # Define input arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True,
                    help="path to input directory of images")
    ap.add_argument("-m", "--model", required=True,
                    help="path to the model")
    args = vars(ap.parse_args())

    # Load the model and the label binarizer
    model, lb = import_model(args["model"])

    # loop over image paths
    image_paths = list(paths.list_images(args["input"]))
    for image_path in image_paths:
        image, predictions = predict_captcha_image(image_path, lb)
        # show the output image
        print("[INFO] captcha: {}".format("".join(predictions)))
        cv2.imshow("Output", image)
        cv2.waitKey()


def import_model(model_path):
    # load the pre-trained network and label binarizer
    print("[INFO] loading pre-trained network...")
    model = load_model(model_path)
    lb = load(open(os.path.join(model_path, "label_binarizer.pkl"), 'rb'))
    return model, lb


def predict_captcha_image(image_path, model, lb):
    # load the image, extract the characters, preprocess them and predict
    image = cv2.imread(image_path)
    image = cv2.copyMakeBorder(image, 25, 0, 0, 0,
                               cv2.BORDER_CONSTANT, value=(255, 255, 255))
    image_characters, bounding_boxes = extract_the_characters(image)
    predictions = []

    for c, b in zip(image_characters, bounding_boxes):
        # pre-process the character and classify it
        roi = preprocess(c, 28, 28)
        roi = np.expand_dims(img_to_array(roi), axis=0) / 255.0
        pred = model.predict(roi).argmax(axis=1)[0]
        pred = lb.classes_[pred]
        predictions.append(str(pred))

        (x, y, w, h) = b
        # draw the prediction on the output image
        cv2.rectangle(image, (x - 2, y - 2),
                      (x + w + 4, y + h + 4), (0, 0, 255), 1)
        cv2.putText(image, str(pred), (x - 5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
    predictions = "".join(predictions)
    return image, predictions


if __name__ == '__main__':
    run()
