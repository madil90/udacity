from keras.models import Sequential
from keras.layers import Input, Dense, Flatten, Lambda, Conv2D, MaxPooling2D, Cropping2D, Dropout
import argparse
import os
import cv2
import csv
import numpy as np
from sklearn.model_selection import train_test_split
import sklearn
from tools import threadsafe_generator
import tensorflow as tf
import math

# create input params
parser = argparse.ArgumentParser(description='Process some integers')
parser.add_argument('--training_dir',default='../../../../Google-Drive/self-driving-nanodegree/behavioral-learning/training_data/', type=str, help="folder with sub-folders for training data")
parser.add_argument('--model_name', default='../../../../Google-Drive/self-driving-nanodegree/behavioral-learning/models/modelv0', help="Model name for saving model")
args = parser.parse_args()
exclude_dirs = ['first_run', 'recovery_driving', 'turns', 'reverse_driving']

# selecting which gpu to use 
os.environ["CUDA_VISIBLE_DEVICES"]="1"

# load csv file first
lines = []
dirs = os.listdir(args.training_dir)
for dir in dirs:
    if dir in exclude_dirs:
        continue
    # now load the labels for this dir
    label_file_path = os.path.join(args.training_dir, dir, 'driving_log.csv')
    with open(label_file_path, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            lines.append(row)
train_samples, validation_samples = train_test_split(lines, test_size=0.2)
print('No of lines found', len(lines))

# data pre-processing parameters
crop_top = 80

def get_image(csv_image_path):
    filename = os.path.join(*(csv_image_path.split('/')[-3:]))
    filepath = os.path.join(args.training_dir, filename)
    return cv2.imread(filepath)

# generator function for data
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # loop forever (when will this terminate?)
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images=[]
            angles=[]
            for batch_sample in batch_samples:
                center_image = get_image(batch_sample[0])     
                center_angle = float(batch_sample[3])

                # flip images and add
                image_flipped = np.fliplr(center_image)
                angle_flipped = -center_angle

                # add left and right images also
                left_correction = 0.035
                right_correction = 0.035
                angle_left = center_angle + left_correction
                angle_right = center_angle - right_correction
                #img_left = get_image(batch_sample[1])
                #img_right = get_image(batch_sample[2])

                images.extend((center_image, image_flipped))
                angles.extend((center_angle, angle_flipped))
                #images.extend((center_image, img_left, img_right, image_flipped))
                #angles.extend((center_angle, angle_left, angle_right, angle_flipped))

            
            # Do more image processing here (trim image perhaps?)
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# generator function for data generation
def createLenet():
    model=Sequential()
    model.add(Cropping2D(cropping=((50,0), (0,0)), input_shape=(160,320,3)))
    model.add(Lambda(lambda x: tf.image.rgb_to_grayscale(x)))
    model.add(Lambda(lambda x:(x/255.0)-0.5 , input_shape=(160,320,3)))
    model.add(Conv2D(32, kernel_size=(3, 3),
                    activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))

# creates a network and returns a model
def createNetwork():
    keep_prob = 0.8
    model = Sequential()
    model.add(Cropping2D(cropping=((50,0), (0,0)), input_shape=(160,320,3)))
    model.add(Lambda(lambda x: tf.image.rgb_to_grayscale(x)))
    model.add(Lambda(lambda x:(x/255.0)-0.5 , input_shape=(160,320,3)))
    model.add(Conv2D(filters=24, kernel_size=(5,5),
                    strides=(1,1), padding='valid',
                    activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2), padding='same'))
    model.add(Conv2D(filters=36, kernel_size=(5,5),
                    strides=(1,1), padding='valid',
                    activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2), padding='same'))
    model.add(Conv2D(filters=48, kernel_size=(5,5),
                    strides=(1,1), padding='valid',
                    activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2), padding='same'))                    
    model.add(Conv2D(filters=64, kernel_size=(3,3),
                    strides=(1,1), padding='valid',
                    activation='relu'))
    model.add(Flatten())
    model.add(Dropout(keep_prob))
    model.add(Dense(1164))
    model.add(Dropout(keep_prob))
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    #model.add(Lambda(lambda x: 2.0*atan(x)))

    model.compile(loss='mse', optimizer='adam')
    return model

# training params
batch_size = 32
n_epochs = 3

# create the network and data generators
model = createNetwork()
train_generator = generator(train_samples, batch_size=batch_size)
valid_generator = generator(validation_samples, batch_size=batch_size)

model.fit_generator(train_generator, steps_per_epoch=len(train_samples)/batch_size \
                    , validation_data=valid_generator, \
                    validation_steps=len(validation_samples)/batch_size,\
                    epochs=n_epochs, use_multiprocessing=True, max_queue_size=64)
model.save(args.model_name + '.h5')

