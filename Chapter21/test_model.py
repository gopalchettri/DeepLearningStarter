# Import the necessary packages
from tensorflow.keras.models import load_model
from utilities.utils.captchahelper import preprocess
from tensorflow.keras.preprocessing.image import img_to_array
import imutils
from imutils import paths
from imutils import contours
import argparse
import numpy as np
import cv2

# Construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True, help="path to input directory of images")
ap.add_argument("-m", "--model", required=True, help="path to input model")
args = vars(ap.parse_args())

def gpu_grow_memory():
    import tensorflow as tf
    from distutils.version import LooseVersion
    import warnings
    from tensorflow.compat.v1.keras.backend import set_session
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    config.log_device_placement = True  # to log device placement (on which device the operation ran)
    sess = tf.compat.v1.Session(config=config)
    set_session(sess)
        

    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    if not tf.test.gpu_device_name(): 
        warnings.warn('No GPU found')
    else:
        print('Default GPU device: {}' .format(tf.test.gpu_device_name()))
    

# Load the pre-trained network
print("[INFO]: Loading pre-trained network....")
gpu_grow_memory()
model = load_model(args["model"])

# Radomly sample a few of the input images
image_paths = list(paths.list_images(args["input"]))
image_paths = np.random.choice(image_paths, size=(10,), replace=False)

# Actually breaking the captcha
# Loop over the image paths
for image_path in image_paths:
    # Load the image and construct it to grayscale, then pad the image to ensure digits caught only the border of the
    # image are retained
    image = cv2.imread(image_path)
    # Diplaying the image 
    # cv2.imshow("Base Image",image) # comment if don't want to see the image
    # cv2.waitKey() # comment it
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.copyMakeBorder(gray, 20, 20, 20, 20, cv2.BORDER_REPLICATE)
    # cv2.imshow("Gray Image", gray) # comment if don't want to see the image
    # cv2.waitKey() # comment it

    # Threshold the image to reveal the digits
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # Find contours in the image, keeping only the four largest ones, then sort them from left-to-right
    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[1] if imutils.is_cv2() else contours[0]
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:4]

    # Initialize the output image as a "grayscale" image with 3 channels along with the output predictions
    output = cv2.merge([gray] * 3)
    # cv2.imshow("Final Image",output) # comment if don't want to see the image
    # cv2.waitKey() # comment it
    predictions = []

    # Loop over the contours
    for c in contours:
        # Compute the bounding box for the contour then extract the digit
        (x, y, w, h) = cv2.boundingRect(c)
        roi = gray[y - 5:y + h + 5, x - 5:x + w + 5]

        # Pre-process the ROI and classify it
        roi = preprocess(roi, 28, 28)
        roi = np.expand_dims(img_to_array(roi), axis=0) /255.0
        pred = model.predict(roi).argmax(axis=1)[0] + 1
        predictions.append(str(pred))
        print("prediction :", str(predictions))

        # draw the prediction on the output image
        cv2.rectangle(output, (x - 2, y - 2), (x + w + 4, y + h + 4), (0, 255, 0), 1)
        cv2.putText(output, str(pred), (x - 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
       
    # Show the output image
    print("[INFO]: Captcha: {}".format("".join(predictions)))
    cv2.imshow("Output", output)
    cv2.waitKey()
