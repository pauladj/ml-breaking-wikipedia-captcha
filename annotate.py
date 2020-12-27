import argparse
import os

import cv2
from imutils import paths

from utils.captcha_helper import extract_the_characters

# Define input arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
                help="path to input directory of images")
ap.add_argument("-a", "--annot", required=True,
                help="path to output directory of annotations")
args = vars(ap.parse_args())

# Grab the image paths then initialize the dictionary of character counts
image_paths = list(paths.list_images(args["input"]))
counts = {}


def ask_for_character_key(character_image):
    # Display the character then wait for a keypress
    cv2.imshow("ROI", character_image)
    key = cv2.waitKey(0)
    cv2.destroyWindow("ROI")

    # If the '`' key is pressed, then ignore the character
    if key == ord("`"):
        print("[INFO]: Ignoring character....")
        return

    # Grab the key that was pressed and construct the path the output directory
    key = chr(key).upper()
    dir_path = os.path.sep.join([args["annot"], key])

    # If the output directory does not exist, create it
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # Write the labeled character to file
    count = counts.get(key, 1)
    p = os.path.sep.join([dir_path, "{}.png".format(str(count).zfill(6))])
    cv2.imwrite(p, character_image)

    # Increment the count for the current key
    counts[key] = count + 1


# Loop over the image paths
for (i, image_path) in enumerate(image_paths):
    # Display an update to the user
    print("[INFO]: Processing image {}/{}".format(i + 1, len(image_paths)))

    try:
        # Load the image, convert it to grayscale and extract the characters
        image = cv2.imread(image_path)
        image_characters, _ = extract_the_characters(image)
        for c in image_characters:
            ask_for_character_key(c)
    except:
        print("[INFO]: Skipping image...")
