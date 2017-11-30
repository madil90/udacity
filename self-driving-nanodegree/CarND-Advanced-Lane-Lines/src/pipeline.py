import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np


# Main pipeline for the project
class Pipeline:
    # Perform the full image processing pipeline one by one
    def run_pipeline(self, image):
        self.calculate_calib_matrices()
        return image

    def calculate_calib_matrices(self, calib_folder='../data/camera_cal', draw=False):
        # termination criteria TODO: check what happens if not used
        nx , ny = 9, 6
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((nx*ny,3), np.float32)
        objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)

        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.

        # get all calibration images 
        calib_images = glob.glob(calib_folder+'/*')

        for fname in calib_images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # find the corners
            ret, corners = cv2.findChessboardCorners(gray, (nx,ny), None)

            # if found, add objpoints and image points 
            if ret == True:
                objpoints.append(objp)
                imgpoints.append(corners)

                # Draw and display the corners
                if draw:
                    cv2.drawChessboardCorners(img, (9,6), corners,ret)
                    cv2.imshow('img',img)
                    cv2.waitKey(500)

        
        # now calculate the matrices for calibration
        camera_matrices = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

        # save these for using when distortion happens
        print(len(camera_matrices))
