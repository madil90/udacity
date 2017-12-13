import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import cv2
from line import Line

class LineSearch:
    def __init__(self):
        
        # window settings
        self.window_width = 50 
        self.window_height = 60 # Break image into 9 vertical layers since image height is 720
        self.margin = 100 # How much to slide left and right for searching

        # searching for the line 
        self.line_history = Line()

    def window_mask(self, width, height, img_ref, center,level):
        output = np.zeros_like(img_ref)
        output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height),max(0,int(center-width/2)):min(int(center+width/2),img_ref.shape[1])] = 1
        return output

    def find_window_centroids(self, image, window_width, window_height, margin):
        
        window_centroids = [] # Store the (left,right) window centroid positions per level
        window = np.ones(window_width) # Create our window template that we will use for convolutions
        
        # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
        # and then np.convolve the vertical image slice with the window template 
        
        # Sum quarter bottom of image to get slice, could use a different ratio
        l_sum = np.sum(image[int(3*image.shape[0]/4):,:int(image.shape[1]/2)], axis=0)
        l_center = np.argmax(np.convolve(window,l_sum))-window_width/2
        r_sum = np.sum(image[int(3*image.shape[0]/4):,int(image.shape[1]/2):], axis=0)
        r_center = np.argmax(np.convolve(window,r_sum))-window_width/2+int(image.shape[1]/2)
        
        # Add what we found for the first layer
        window_centroids.append((l_center,r_center))
        
        # Go through each layer looking for max pixel locations
        for level in range(1,(int)(image.shape[0]/window_height)):
            # convolve the window into the vertical slice of the image
            image_layer = np.sum(image[int(image.shape[0]-(level+1)*window_height):int(image.shape[0]-level*window_height),:], axis=0)
            conv_signal = np.convolve(window, image_layer)
            # Find the best left centroid by using past left center as a reference
            # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
            offset = window_width/2
            l_min_index = int(max(l_center+offset-margin,0))
            l_max_index = int(min(l_center+offset+margin,image.shape[1]))
            l_center = np.argmax(conv_signal[l_min_index:l_max_index])+l_min_index-offset
            # Find the best right centroid by using past right center as a reference
            r_min_index = int(max(r_center+offset-margin,0))
            r_max_index = int(min(r_center+offset+margin,image.shape[1]))
            r_center = np.argmax(conv_signal[r_min_index:r_max_index])+r_min_index-offset
            # Add what we found for that layer
            window_centroids.append((l_center,r_center))

        return window_centroids

    def fit_polynomial(self, img, window_centroids, Minv, undist):
        rightx = np.zeros(len(window_centroids))
        leftx = np.zeros(len(window_centroids))
        lefty = np.zeros(len(window_centroids))

        # print(len(window_centroids))

        for level in range(0, len(window_centroids)):
            x_left = window_centroids[level][0]
            y_left = img.shape[0] - (level*self.window_height + self.window_height/2)

            x_right = window_centroids[level][1]

            leftx[level] = x_left
            lefty[level] = y_left
            rightx[level] = x_right

        # adding smoothing here
        self.line_history.add_iteration(leftx, lefty, rightx)

        leftx, lefty, rightx = self.line_history.get_smoothed_line()

        # try to fit a polynomial to these
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(lefty, rightx, 2)

        # new plot here
        # plt.figure()
        # plt.plot(leftx, lefty, 'ro')
        # plt.axis([0, 1200, 720, 0])
        # plt.show()

        # set a new leftx lefty ploty
        ploty = np.arange(0, undist.shape[0], 5)
        left_fitx = np.polyval(left_fit, ploty)
        right_fitx = np.polyval(right_fit, ploty)
        

        # figure out the curvature here also
        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30/720 # meters per pixel in y dimension
        xm_per_pix = 3.7/700 # meters per pixel in x dimension

        # Fit new polynomials to x,y in world space
        y_eval = img.shape[0]
        left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
        right_fit_cr = np.polyfit(lefty*ym_per_pix, rightx*xm_per_pix, 2)
        # Calculate the new radii of curvature
        left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
        # Now our radius of curvature is in meters
        # print(left_curverad, 'm', right_curverad, 'm')
        # Example values: 632.1 m    626.2 m

        # Create an image to draw the lines on
        warp_zero = np.zeros_like(img).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, Minv, (undist.shape[1], undist.shape[0])) 
        # Combine the result with the original image
        result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
        # plt.figure()
        # plt.imshow(result)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(result, 'Left curvature : '+str(left_curverad) , (100,50), font, 1, (255,255,255),2, cv2.LINE_AA)

        return left_fit, right_fit, result

    def search(self, warped):
        window_centroids = self.find_window_centroids(warped, self.window_width, self.window_height, self.margin)
        #print(window_centroids)

        # If we found any window centers
        if len(window_centroids) > 0:

            # Points used to draw all the left and right windows
            l_points = np.zeros_like(warped)
            r_points = np.zeros_like(warped)

            # Go through each level and draw the windows 	
            for level in range(0,len(window_centroids)):
                # Window_mask is a function to draw window areas
                l_mask = self.window_mask(self.window_width,self.window_height,warped,window_centroids[level][0],level)
                r_mask = self.window_mask(self.window_width,self.window_height,warped,window_centroids[level][1],level)
                # Add graphic points from window mask here to total pixels found 
                l_points[(l_points == 255) | ((l_mask == 1) ) ] = 255
                r_points[(r_points == 255) | ((r_mask == 1) ) ] = 255

            # Draw the results
            template = np.array(r_points+l_points,np.uint8) # add both left and right window pixels together
            zero_channel = np.zeros_like(template) # create a zero color channel
            template = np.array(cv2.merge((zero_channel,template,zero_channel)),np.uint8) # make window pixels green
            warpage= np.dstack((warped, warped, warped))*255 # making the original road pixels 3 color channels
            output = cv2.addWeighted(warpage.astype(np.uint8), 1, template, 0.5, 0.0) # overlay the orignal road image with window results

        # If no window centers found, just display orginal road image
        else:
            output = np.array(cv2.merge((warped,warped,warped)),np.uint8)

        # Display the final results
        # plt.figure()
        # plt.imshow(output)
        # plt.title('window fitting results')
        # plt.show()

        return window_centroids, output
