#!/usr/bin/env python3
"""
Module to test with our own images
"""
import numpy as np
from PIL import Image

def image_resize(picture_name):
    """
    Resizes the image according to the right format for the Neural Network
    """
    test_picture = Image.open("../test_images/{}".format(picture_name)).convert('L')   # opens and converts the image in black and white , ecach pixels is an int between 0 (black) and 255 (white)
    test_picture = test_picture.resize((28, 28), Image.ANTIALIAS)         # resiizes the image, that needs to be square first / the rotate(-90) only exists if its a phone photo, skip it otherwise
    test_picture.save("../test_images/{}_resized.png".format(picture_name[:-4]))       # saves the new image in the same folder

def own_test(net, picture_name_resized):
    """
    outputs the prediction for the resized image given
    """
    test_picture = Image.open("../test_images/{}".format(picture_name_resized))        # opens the new resized image
    x = np.reshape(np.array(test_picture).astype(float), (784,1))                     # reshapes it into an (784, 1) array
    test = [[x[i][0]/255] for i in range(784)]                                    # converts all its pixels into floats bewteen 0 (white) and 1 (black)
    return net.own_test(test)
