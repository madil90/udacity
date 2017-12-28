import os
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
import numpy as np
from skimage.feature import hog


# This class handles the training of a classifier
# for Vehicle detection. TODO:
# 1) Make a list of all data (vehicle and non-vehicle)
# 2) Go through list -> process each image -> turn into features
# 3) Store the list of features
# 4) Use the list to train a classifier
class VehicleTrainer:
    def __init__(self):
        # parameters
        self.hog_orientations = 9
        self.hog_pixel_per_cell = 8
        self.hog_cell_per_block = 2
        self.car_list = []
        self.notcar_list = []
        
        # algorithm steps
        self.load_training_set()
        self.extract_features()
    
    # Step 1 of the algorithm
    def load_training_set(self):

        # Step 1a) Read all data
        rootdir = '../data/'
        topdirs = ['vehicles', 'non-vehicles']
        exten = '.png'
        for topdir in topdirs:
            for dirpath, dirnames, files in os.walk(os.path.join(rootdir, topdir)):
                for name in files:
                    if name.lower().endswith(exten):
                        filepath = self.car_list.append(os.path.join(dirpath, name))
                        if topdir is 'vehicles':
                            self.car_list.append(filepath)
                        else:
                            self.notcar_list.append(filepath)

        # Step 1b) Create a datalookup dict
        data_dict = {}
        data_dict["n_cars"] = len(self.car_list)
        data_dict["n_notcars"] = len(self.notcar_list)
        example_img = mpimg.imread(self.car_list[0])
        data_dict["image_shape"] = example_img.shape
        data_dict["data_type"] = example_img.dtype
        print(data_dict)

    # Step 2 of the algorithm
    def extract_features(self):
        # start going through car images and extract hog features
        for carpath in self.car_list:
            carimg = mpimg.imread(carpath)
            cargray = cv2.cvtColor(carimg, cv2.COLOR_RGB2GRAY)
            features, hog_image = self.get_hog_features(cargray, 
                                    self.hog_orientations,
                                    self.hog_pixel_per_cell,
                                    self.hog_cell_per_block,
                                    vis=True, feature_vec=False)
            plt.figure()
            plt.imshow(hog_image, cmap='gray')
            plt.title('HOG visualization')
            plt.waitforbuttonpress()
            break

    
    ### Helper functions for the class ###
    # Define a function to return HOG features and visualization
    def get_hog_features(self, img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
        if vis == True:
            features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                    cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False, 
                                    visualise=True, feature_vector=False)
            return features, hog_image
        else:      
            features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                        cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False, 
                        visualise=False, feature_vector=feature_vec)
            return features