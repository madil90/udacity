import os
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
import numpy as np
import time
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# This class handles the training of a classifier
# for Vehicle detection. TODO:
# 1) Make a list of all data (vehicle and non-vehicle)
# 2) Go through list -> process each image -> turn into features
# 3) Store the list of features
# 4) Use the list to train a classifier