import argparse
import os
import time

import requests

# Define input arguments
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True,
                help="path to output directory of images")
ap.add_argument("-n", "--num_images", type=int, default=500,
                help="# of images to download")
args = vars(ap.parse_args())

# Initialize the URL that returns the new captcha id
url_id = "https://es.wikipedia.org/w/api.php"
data = {"action": "fancycaptchareload", "format": "json"}

# Initialize the URL that returns the new captcha by its id
url_image = "https://es.wikipedia.org/w/index.php?title=Especial:Captcha/" \
            "image&wpCaptchaId="

total = 0

# Loop over the number of images to download
for i in range(0, args["num_images"]):
    try:
        # Request new captcha id
        r = requests.post(url_id, timeout=60, data=data)
        if r.status_code != 200:
            break

        response = r.json()
        captcha_id = response["fancycaptchareload"]["index"]

        # Request captcha image
        r = requests.get(url_image + captcha_id, timeout=60)

        # Save the image to disk
        p = os.path.sep.join(
            [args["output"], "{}.jpg".format(str(total).zfill(5))])
        f = open(p, "wb")
        f.write(r.content)
        f.close()

        # Update the counter
        print("[INFO]: Downloaded: {}".format(p))
        total += 1

    # Handle if any exceptions are thrown during the download process
    except:
        print("[INFO]: Error downloading image....")

    # Insert a small sleep to be courteous to the server
    time.sleep(0.1)
