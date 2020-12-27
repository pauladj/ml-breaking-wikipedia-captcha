# import the necessary packages
import cv2
import imutils
import numpy as np


def preprocess(image, width, height):
    # grab the dimensions of the image, then initialize
    # the padding values
    (h, w) = image.shape[:2]

    # if the width is greater than the height then resize along
    # the width
    if w > h:
        image = imutils.resize(image, width=width)

    # otherwise, the height is greater than the width so resize
    # along the height
    else:
        image = imutils.resize(image, height=height)

    # determine the padding values for the width and height to
    # obtain the target dimensions
    padW = int((width - image.shape[1]) / 2.0)
    padH = int((height - image.shape[0]) / 2.0)

    # pad the image then apply one more resizing to handle any 
    # rounding issues
    image = cv2.copyMakeBorder(image, padH, padH, padW, padW,
                               cv2.BORDER_CONSTANT, value=255)
    image = cv2.resize(image, (width, height))

    # return the pre-processed image 
    return image


def sort_contours(cnts, method="left-to-right"):
    # initialize the reverse flag and sort index
    reverse = False
    i = 0

    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True

    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1

    # construct the list of bounding boxes and sort them from top to
    # bottom
    bounding_boxes = [cv2.boundingRect(c) for c in cnts]
    cnts, bounding_boxes = zip(*sorted(zip(cnts, bounding_boxes),
                                       key=lambda b: b[1][i],
                                       reverse=reverse))

    # return the list of sorted contours and bounding boxes
    return cnts, bounding_boxes


def extract_the_characters(image):
    """
    Given a captcha image extract all the characters
    :param image: The captcha image
    :return: list with the individual characters and bounding boxes
    """
    individual_characters = []
    bounding_boxes = []  # of every individual character

    # convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = 255 - gray  # inverse it

    # Blur it and binarize the image {0,1}
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    (T, thresholded) = cv2.threshold(blurred, 140, 255, cv2.THRESH_BINARY)

    # OTSU
    (T, thresh_contours) = cv2.threshold(gray, 0, 255,
                                         cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Find the external contours and sort them from left to right using the
    # binarized image using OTSU
    cnts, cnts_hierarchy = cv2.findContours(thresh_contours.copy(),
                                            cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_SIMPLE)
    cnts, boxes = sort_contours(cnts, method="left-to-right")

    i_done = -1  # which contours are already added to one of the characters
    for i, c in enumerate(cnts):
        # for every contour
        clone = image.copy()

        # area
        area = cv2.contourArea(c)
        if area < 5:
            # just a point
            continue

        if i_done >= i:
            # if the current contour is already added continue to the next one
            continue

        # bounding box
        (x, y, w, h) = boxes[i]
        cv2.rectangle(clone, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # extract character from current contour
        character_mask = np.zeros(gray.shape, dtype="uint8")
        cv2.drawContours(character_mask, [c], -1, 255, -1)

        # check if to add the next contour(s) to the current one
        # for example, the dot of the i
        add_next = True
        i_done = i
        while add_next:
            add_next = False
            if i_done < len(cnts) - 1:
                # If there's a next contour get its bounding box
                (next_x, next_y, next_w, next_h) = boxes[i_done + 1]
                if x < next_x < (x + w) and (next_w < 15):
                    # If the next contour is above or below add it to
                    # the current character
                    add_next = True
                    i_done += 1
                    cv2.drawContours(character_mask, [cnts[i_done]], -1, 255, -1)
                else:
                    add_next = False

        s = cv2.bitwise_and(thresholded, character_mask)
        (x, y, w, h) = cv2.boundingRect(s)
        character = 255 - s[y:h + y, x:x + w]

        if w > 65 and area > 300:
            # three character together, separate them
            one_char_width = int(character.shape[1] / 3)
            left_char = character[:, :one_char_width]
            individual_characters.append(left_char)
            bounding_boxes.append((x, y, one_char_width, h))

            center_char = character[:, one_char_width:one_char_width * 2]
            individual_characters.append(center_char)
            bounding_boxes.append((x+one_char_width, y, one_char_width, h))

            right_char = character[:, one_char_width * 2:]
            individual_characters.append(right_char)
            bounding_boxes.append((x+one_char_width*2, y, one_char_width, h))
        elif w >= 37 and area > 300:
            # two character together, separate them in the middle
            one_char_width = int(character.shape[1] / 2)
            left_char = character[:, :one_char_width]
            individual_characters.append(left_char)
            bounding_boxes.append((x, y, one_char_width, h))

            right_char = character[:, one_char_width:]
            individual_characters.append(right_char)
            bounding_boxes.append(
                (x + one_char_width, y, one_char_width, h))
        else:
            # just one character
            individual_characters.append(character)
            bounding_boxes.append((x, y, w, h))

    return individual_characters, bounding_boxes
