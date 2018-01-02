import os
from vehicle_trainer import VehicleTrainer

# Main class that trains a classifier for 
# vehicles and predicts using the classifier
class VehicleTracker:
    def __init__(self):
        # TODO:
        # see if anything should be done here
        pass
    
    # train the detector
    def train_detector(self):
        # TODO:
        # is this really needed? only calling the class method
        # currently creates and trains a model for vehicle detection
        self.vehicle_trainer = VehicleTrainer()
    
    # vehicle detector using windows
    def detect_vehicles(self):
        # TODO :
        # 1) load video or images 
        # 2) Get features
        # 3) Run windows 
        # 4) overlay detections
        pass


# Very simple main method for launching our main Vehicle Detector class
if __name__=='__main__':
    vehicle_detector = VehicleTracker()
    vehicle_detector.train_detector()