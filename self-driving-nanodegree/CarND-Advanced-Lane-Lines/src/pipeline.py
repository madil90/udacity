import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from thresholding import Thresholding
from perspective import Perspective
from line_search import LineSearch

# Main pipeline for the project
class Pipeline:
    # constructor
    def __init__(self):
        self.mtx_file_path = 'camera_matrices/mtx.npz'
        self.camera_matrices = None
        self.thresholding_pipeline = Thresholding()
        self.perspective = Perspective()
        self.load_camera_matrices()
        self.line_search = LineSearch()

    # Perform the full image processing pipeline one by one
    def run_pipeline(self, image):
        self.calculate_calib_matrices()
        dist = self.undistort_image(image)
        thresh_image = self.thresholding_pipeline.process_image(dist)
        pers_image = self.perspective.warp_image(thresh_image)
        centroids,output = self.line_search.search(pers_image)
        left_fit, right_fit, result = self.line_search.fit_polynomial(pers_image, centroids, self.perspective.pers_Minv, dist)

        
        # print undistorted image
        # plt.figure()
        # plt.imshow(dist)
        # plt.waitforbuttonpress()

        # plt.figure()
        plt.imshow(pers_image, cmap='gray')
        plt.waitforbuttonpress()

        plt.imshow(result, cmap='gray')
        plt.waitforbuttonpress()

        return result

    def calculate_calib_matrices(self, calib_folder='../data/camera_cal', draw=False):
        # check if we have the matrices
        if self.camera_matrices is not None:
            return

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
                    cv2.waitKey(0)

        
        # now calculate the matrices for calibration
        ret, mtx, dist, revcs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

        # save these for using when distortion happens
        np.savez(self.mtx_file_path, mtx=mtx, dist=dist)

        self.camera_matrices = (mtx, dist)
    
    def load_camera_matrices(self):
        if os.path.isfile(self.mtx_file_path):
            npzfile = np.load(self.mtx_file_path)
            mtx, dist = npzfile['mtx'], npzfile['dist']
            self.camera_matrices = (mtx, dist)

    def undistort_image(self, image):
        mtx, dist = self.camera_matrices
        h, w = image.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

        # undistort
        dst = cv2.undistort(image, mtx, dist, None, newcameramtx)
        x,y,w,h = roi
        dst = dst[y:y+h, x:x+w]
        return dst
