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
class VehicleTrainer:
    def __init__(self):
        # parameters for hog
        self.hog_orientations = 9
        self.hog_pixel_per_cell = 8
        self.hog_cell_per_block = 2

        # parameters for color features
        self.color_spatial = 32
        self.color_histbin = 32

        # list initialization
        self.car_list = []
        self.notcar_list = []
        
        # algorithm steps
        self.car_list, self.notcar_list, self.data_dict = self.load_training_set()
        self.data_X, self.labels_y = self.extract_features()
        self.model = self.train_model(data_X, labels_y)
    
    # Step 1 of the algorithm
    def load_training_set(self):

        # Step 1a) Read all data
        car_list = []
        notcar_list = []
        rootdir = '../data/'
        topdirs = ['vehicles', 'non-vehicles']
        exten = '.png'
        for topdir in topdirs:
            for dirpath, dirnames, files in os.walk(os.path.join(rootdir, topdir)):
                for name in files:
                    if name.lower().endswith(exten):
                        filepath = os.path.join(dirpath, name)
                        if topdir is 'vehicles':
                            car_list.append(filepath)
                        else:
                            notcar_list.append(filepath)

        # Step 1b) Create a datalookup dict
        data_dict = {}
        data_dict["n_cars"] = len(car_list)
        data_dict["n_notcars"] = len(notcar_list)
        example_img = mpimg.imread(car_list[0])
        data_dict["image_shape"] = example_img.shape
        data_dict["data_type"] = example_img.dtype
        print(data_dict)

        return car_list, notcar_list, data_dict

    # Step 2 of the algorithm
    def extract_features(self):
        
        # try using color features only first
        car_features = self.extract_hog_color_features(self.car_list, cspace='RGB', spatial_size=(self.color_spatial, self.color_spatial),
                        hist_bins=self.color_histbin, hist_range=(0, 256))
        notcar_features = self.extract_hog_color_features(self.notcar_list, cspace='RGB', spatial_size=(self.color_spatial, self.color_spatial),
                        hist_bins=self.color_histbin, hist_range=(0, 256))
        
        # Create an array stack of feature vectors
        X = np.vstack((car_features, notcar_features)).astype(np.float64)
        X_scaler = StandardScaler().fit(X)
        data = X_scaler.transform(X) # originally called scaled_X

        # Define the labels vector
        labels = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

        return data, labels

    # Step 3 : Train a model 
    def train_model(self, data, labels):
        # Split up data into randomized training and test sets
        rand_state = np.random.randint(0, 100)
        X_train, X_test, y_train, y_test = train_test_split(
            data, labels, test_size=0.1, random_state=rand_state)

        print('Using spatial binning of:',self.color_spatial,
            'and', self.color_histbin,'histogram bins')
        print('Feature vector length:', len(X_train[0]))
        # Use a linear SVC 
        svc = LinearSVC()
        # Check the training time for the SVC
        t=time.time()
        svc.fit(X_train, y_train)
        t2 = time.time()
        print(round(t2-t, 2), 'Seconds to train SVC...')
        # Check the score of the SVC
        print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
        # Check the prediction time for a single sample
        t=time.time()
        n_predict = 10
        print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
        print('For these',n_predict, 'labels: ', y_test[0:n_predict])
        t2 = time.time()
        print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')
        return svc
    
    ################################################################
    ############# Helper functions for the class ###################
    ################################################################

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

    # Define a function to compute binned color features  
    def bin_spatial(self, img, size=(32, 32)):
        # Use cv2.resize().ravel() to create the feature vector
        features = cv2.resize(img, size).ravel() 
        # Return the feature vector
        return features

    # Define a function to compute color histogram features  
    def color_hist(self, img, nbins=32, bins_range=(0, 256)):
        # Compute the histogram of the color channels separately
        channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
        channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
        channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
        # Concatenate the histograms into a single feature vector
        hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
        # Return the individual histograms, bin_centers and feature vector
        return hist_features

    # Define a function to extract features from a list of images
    # Have this function call bin_spatial() and color_hist()
    def extract_hog_color_features(self, imgs,  
                            orient =9, pixels_per_cell=8, cells_per_block=2,
                            cspace='RGB', spatial_size=(32, 32),
                            hist_bins=32, hist_range=(0, 256),
                            hog_enabled=True, color_enabled=True):
        # Create a list to append feature vectors to
        features = []
        # Iterate through the list of images
        for file in imgs:
            # Read in each one by one
            image = mpimg.imread(file)
            # apply color conversion if other than 'RGB'
            if cspace != 'RGB':
                if cspace == 'HSV':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
                elif cspace == 'LUV':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
                elif cspace == 'HLS':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
                elif cspace == 'YUV':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            else: feature_image = np.copy(image)      
            
            # Apply bin_spatial() to get spatial color features
            spatial_features = self.bin_spatial(feature_image, size=spatial_size)
            # Apply color_hist() also with a color space option now
            hist_features = self.color_hist(feature_image, nbins=hist_bins, bins_range=hist_range)
            # Apply hog() for hog_features
            hog_features = self.get_hog_features(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), 
                                                orient=orient, pix_per_cell=pixels_per_cell, cell_per_block=cells_per_block)

            if color_enabled and hog_enabled:
                features_array = np.concatenate((hog_features, spatial_features, hist_features))
            elif color_enabled and not hog_enabled:
                features_array = np.concatenate((spatial_features, hist_features))
            elif hog_enabled and not color_enabled:
                features_array = hog_features
            
            # Append the new feature vector to the features list
            features.append(features_array)
        # Return list of feature vectors
        return features