import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

class Thresholding:
    def __init__(self):
        pass
    
    def process_image(self, image, draw=True):
        # Choose a Sobel kernel size
        ksize = 3 # Choose a larger odd number to smooth gradient measurements

        # Apply each of the thresholding functions
        gradx = self.abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=(20, 80))
        grady = self.abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh=(15, 80))
        mag_binary = self.mag_thresh(image, sobel_kernel=ksize, mag_thresh=(30, 100))
        dir_binary = self.dir_threshold(image, sobel_kernel=ksize, thresh=(0.397, 1.17))

        combined = np.zeros_like(dir_binary)
        combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1

        # perhaps draw these images and save for testing? 
        if draw:
            fig = plt.figure()
            a = fig.add_subplot(2,3,1)
            imgplot = plt.imshow(gradx, cmap='gray')
            a.set_title('Gradient x')

            a = fig.add_subplot(2,3,4)
            imgplot = plt.imshow(grady, cmap='gray')
            a.set_title('Gradient y')

            a = fig.add_subplot(2,3,2)
            imgplot = plt.imshow(mag_binary, cmap='gray')
            a.set_title('Magnitude')

            a = fig.add_subplot(2,3,5)
            imgplot = plt.imshow(dir_binary, cmap='gray')
            a.set_title('Direction')

            a = fig.add_subplot(2,3,(3,6))
            imgplot = plt.imshow(combined, cmap='gray')
            a.set_title('Combined')


            plt.show()

        return combined

    def abs_sobel_thresh(self, img, orient='x', sobel_kernel=3, thresh=(0, 255)):
        # Calculate directional gradient
        # Apply threshold
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # Apply x or y gradient with the OpenCV Sobel() function
        # and take the absolute value
        if orient == 'x':
            abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
        if orient == 'y':
            abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
        # Rescale back to 8 bit integer
        scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
        # Create a copy and apply the threshold
        binary_output = np.zeros_like(scaled_sobel)
        # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
        binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
        return binary_output

    def mag_thresh(self, img, sobel_kernel=3, mag_thresh=(0, 255)):
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # Take both Sobel x and y gradients
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        # Calculate the gradient magnitude
        gradmag = np.sqrt(sobelx**2 + sobely**2)
        # Rescale to 8 bit
        scale_factor = np.max(gradmag)/255 
        gradmag = (gradmag/scale_factor).astype(np.uint8) 
        # Create a binary image of ones where threshold is met, zeros otherwise
        binary_output = np.zeros_like(gradmag)
        binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1
        return binary_output

    def dir_threshold(self, img, sobel_kernel=3, thresh=(0, np.pi/2)):
        # Grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # Calculate the x and y gradients
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        # Take the absolute value of the gradient direction, 
        # apply a threshold, and create a binary image result
        absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
        binary_output =  np.zeros_like(absgraddir)
        binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1
        return binary_output