import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# perspective transform and dewarping
class Perspective:
    def __init__(self):
        self.pers_M = None
        self.pers_Minv = None
        #self.src_points = np.float32([ [501,523], [501,765], [662,1015], [662,287]])
        #self.dest_points = np.float32([ [501,287], [501,1015], [662,1015], [662,287] ])
        #self.src_points = np.float32([ [523,501], [765,501], [1015,662], [287,662]])
        #self.dest_points = np.float32([ [500,400], [765,400], [765,662], [500,662] ])
        # self.src_points = np.float32([ [197,717], [581,461], [701,461], [1098,717]])
        # self.dest_points = np.float32([ [475,717], [475,0], [835,0], [835,717] ])
        # self.src_points = np.float32([ [195,720], [590,400], [702,400], [1120,720]])
        # self.dest_points = np.float32([ [320,720], [320,0], [930,0], [930,720] ])
        self.src_points = np.float32(
            [[300, 620],
            [539,413],
            [687, 413],
            [1087, 620]]
        )

        self.dest_points = np.float32(
            [[300, 620],
            [300, 0],
            [1000, 0],
            [1000, 620]]
        )
        self.perspective_image = '../data/test_images/straight_lines1.jpg'
        self.calculate_matrix()
        #self.plot_points()
        pass

    def warp_image(self, image):
        img_size = (image.shape[1], image.shape[0])
        return cv2.warpPerspective(image, self.pers_M, img_size, flags=cv2.INTER_LINEAR)

    def calculate_matrix(self):
        self.pers_M = cv2.getPerspectiveTransform(self.src_points, self.dest_points)

        self.pers_Minv = cv2.getPerspectiveTransform(self.dest_points, self.src_points)

        
        # warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
        # use this for warning

    def plot_points(self):
        image = mpimg.imread(self.perspective_image)
        # plt.imshow(image)
        # for point in self.src_points:
        #     plt.plot(point[0],point[1],'.')
        # plt.waitforbuttonpress()
        

