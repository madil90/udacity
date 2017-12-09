# Loads images and passes to pipeline
from pipeline import Pipeline
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def toRGB(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

if __name__=='__main__':
    cap = cv2.VideoCapture('../data/project_video.mp4')

    while(cap.isOpened()):
        ret, img = cap.read()

        # test image is 'calibration2.jpg' right now
        plt.ion()
        # img = cv2.imread('../data/test_images/test6.jpg')
        # if img is None:
        #     print('Image not found')
        #     exit()


        pipeline = Pipeline()
        processed_img = pipeline.run_pipeline(img)

        # Draw both images together
        plt.figure(0)
        plt.subplot(1,2,1)
        plt.imshow(toRGB(img))
        plt.title('Original')
        plt.subplot(1,2,2)
        plt.imshow(toRGB(processed_img))
        plt.title('Processed')
        plt.pause(0.05)
    
    cap.release()
    cv2.destroyAllWindows()
