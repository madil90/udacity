# Load pickled data
import pickle
import os
import sys
import numpy as np
from PIL import Image
import PIL

# TODO: Fill this in based on where you saved the training and testing data

base_path = '/Users/deadman/Google Drive/Udacity Self Driving/Traffic Sign Classifier/CarND-Traffic-Sign-Classifier-Project/traffic-signs-data'
training_file = os.path.join(base_path, 'train.p')
validation_file= os.path.join(base_path, 'valid.p')
testing_file = os.path.join(base_path, 'test.p')



with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

### Replace each question mark with the appropriate value. 
### Use python, pandas or numpy methods rather than hard coding the results

# TODO: Number of training examples
n_train = len(X_train)

# TODO: Number of validation examples
n_validation = len(X_valid)

# TODO: Number of testing examples.
n_test = len(X_test)

# TODO: What's the shape of an traffic sign image?
image_shape = X_train[0].shape

# TODO: How many unique classes/labels there are in the dataset.
n_classes = len(set(y_train))

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

### Data exploration visualization code goes here.
### Feel free to use as many code cells as needed.
import matplotlib.pyplot as plt
# Visualizations will be shown in the notebook.
#%matplotlib inline
from collections import defaultdict
import numpy as np

# plot a random image
plt.imshow(X_train[2500])

# plot the count of each sign
class_dict = defaultdict(int)
for label in y_train:
    # put in a dictionary
    class_dict[label] += 1

class_freqs = list(class_dict.values())
n_bins = len(class_dict.keys())
print(np.asarray(class_freqs))

#fig, ax = plt.subplots()
#ax.plot(range(0,n_bins), class_freqs ,'ro')
#ax.hist(np.asarray(class_freqs), n_bins)
#plt.show()



### Preprocess the data here. It is required to normalize the data. Other preprocessing steps could include 
### converting to grayscale, etc.
### Feel free to use as many code cells as needed.

# Pre-processing data here

# convert images to different channel here (lets do grayscale first)
def rgb2gray(img):
    return np.dot(img[...,:3], [0.299, 0.587, 0.114])

# X_train_gray = np.empty(X_train.shape[:3])
# X_valid_gray = np.empty(X_valid.shape[:3])
# X_test_gray = np.empty(X_test.shape[:3])
# # print(X_train.shape[:3])
# # print(X_train_gray.shape)
# for i in range(0, len(X_train)):
#     X_train_gray[i] = rgb2gray(X_train[i])
# for i in range(0, len(X_valid)):
#     X_valid_gray[i] = rgb2gray(X_valid[i])
# for i in range(0, len(X_test)):
#     X_test_gray[i] = rgb2gray(X_test[i])

# # expanding dimensions for compatibility
# X_train_gray = np.expand_dims(X_train_gray, axis=3)
# X_valid_gray = np.expand_dims(X_valid_gray, axis=3)
# X_test_gray = np.expand_dims(X_test_gray, axis=3)

# print('Length of training : ', len(X_train) == len(X_train_gray))
# print('Length of validation : ', len(X_valid) == len(X_valid_gray))
# print('Length of testing : ', len(X_test) == len(X_test_gray))

# TODO: Remove this later
# sys.exit()

# normalzie the data
# convert image to YCbCr
from PIL import Image
def getYCbCr(rgb):
    img = Image.fromarray(rgb)
    img_yuv = img.convert('YCbCr')
    return img_yuv

# for i in range(0, len(X_train)):
#     X_train[i] = getYCbCr(X_train[i])
# for i in range(0, len(X_valid)):
#     X_valid[i] = getYCbCr(X_valid[i])
# for i in range(0, len(X_test)):
#     X_test[i] = getYCbCr(X_test[i])

def normalize_img(img):
    return (img-128.0)/128.0


# for i in range(0, len(X_train)):
#     X_train[i] = (X_train[i]-128.0)/128.0

X_train = normalize_img(X_train)
X_valid = normalize_img(X_valid)
X_test = normalize_img(X_test)

# print('printing next one')
# print(X_train[0])

def shuffle_two(x, y):
    s = np.arange(x.shape[0])
    np.random.shuffle(s)
    return x[s], y[s]


### Define your architecture here.
### Feel free to use as many code cells as needed.

import tensorflow as tf
from tensorflow.contrib.layers import flatten

EPOCHS = 10
BATCH_SIZE = 128
n_channels = 3 # grayscale or rgb used


def LeNet(x):    
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1
    dropout_prob = 0.75
    
    # SOLUTION: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, n_channels, 16), mean = mu, stddev = sigma))
    conv1_b = tf.Variable(tf.zeros(16))
    conv1   = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b

    # SOLUTION: Activation.
    conv1 = tf.nn.relu(conv1)

    # SOLUTION: Pooling. Input = 28x28x6. Output = 14x14x6.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID') 

    # SOLUTION: Layer 2: Convolutional. Output = 10x10x16.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 16, 32), mean = mu, stddev = sigma))
    conv2_b = tf.Variable(tf.zeros(32))
    conv2   = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
    
    # SOLUTION: Activation.
    conv2 = tf.nn.relu(conv2)

    # SOLUTION: Pooling. Input = 10x10x16. Output = 5x5x16.
    #conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # add additional convolution layer with strides of 2
    conv3_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 32, 64), mean = mu, stddev = sigma))
    conv3_b = tf.Variable(tf.zeros(64))
    conv3   = tf.nn.conv2d(conv2, conv3_W, strides=[1, 2, 2, 1], padding='VALID') + conv3_b

    conv3 = tf.nn.relu(conv3)

    # SOLUTION: Flatten. Input = 5x5x16. Output = 400.
    fc0   = flatten(conv3)
    
    # SOLUTION: Layer 3: Fully Connected. Input = 400. Output = 120.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(576, 400), mean = mu, stddev = sigma))
    fc1_b = tf.Variable(tf.zeros(400))
    fc1   = tf.matmul(fc0, fc1_W) + fc1_b
    
    # SOLUTION: Activation.
    fc1    = tf.nn.relu(fc1)

    # add dropout layer
    drp1 = tf.nn.dropout(fc1, keep_prob=dropout_prob)

    # SOLUTION: Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2_W  = tf.Variable(tf.truncated_normal(shape=(400, 120), mean = mu, stddev = sigma))
    fc2_b  = tf.Variable(tf.zeros(120))
    fc2    = tf.matmul(drp1, fc2_W) + fc2_b
    
    # SOLUTION: Activation.
    fc2    = tf.nn.relu(fc2)

    # SOLUTION: Layer 5: Fully Connected. Input = 84. Output = 10.
    fc3_W  = tf.Variable(tf.truncated_normal(shape=(120, n_classes), mean = mu, stddev = sigma))
    fc3_b  = tf.Variable(tf.zeros(n_classes))
    logits = tf.matmul(fc2, fc3_W) + fc3_b
    
    return logits


# module for evaluation
x = tf.placeholder(tf.float32, (None, 32, 32, n_channels))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, n_classes)

# adding learning rate with decay
global_step = tf.Variable(0, trainable=False)
start_learning_rate = 0.001
decayed_learning_rate = tf.train.exponential_decay(start_learning_rate, global_step,
                                           1000000, 0.90, staircase=True)
rate = 0.001

logits = LeNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation, global_step=global_step)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data, sess):
    num_examples = len(X_data)
    total_accuracy = 0
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

### Train your model here.
### Calculate and report the accuracy on the training and validation set.
### Once a final model architecture is selected, 
### the accuracy on the test set should be calculated and reported as well.
### Feel free to use as many code cells as needed.

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

sess.run(tf.global_variables_initializer())
num_examples = len(X_train)

print("Training...")
print()
for i in range(EPOCHS):
    X_train, y_train = shuffle_two(X_train, y_train)
    for offset in range(0, num_examples, BATCH_SIZE):
        end = offset + BATCH_SIZE
        batch_x, batch_y = X_train[offset:end], y_train[offset:end]
        sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})

    validation_accuracy = evaluate(X_valid, y_valid, sess)
    print("EPOCH {} ...".format(i+1))
    print("Validation Accuracy = {:.3f}".format(validation_accuracy))


# check accuracy on test set 
test_accuracy = evaluate(X_test, y_test, sess)
print("Test Accuracy at end = {:.3f}".format(test_accuracy))

# testing on the web_examples images
# load the web examples and create a dataset 
web_examples = './web_examples'
signs_dir = os.listdir(web_examples)
test_imgs = []
test_classes = []
for directory in signs_dir:
    img_paths = os.listdir(os.path.join(web_examples, directory))
    for img_path in img_paths:
        im = Image.open(os.path.join(web_examples, directory, img_path))
        res = im.resize((32,32), resample=PIL.Image.BILINEAR)
        im_arr = np.asarray(res)
        im_class = int(directory)
        test_classes.append(im_class)
        test_imgs.append(im_arr)

test_imgs = np.asarray(test_imgs)
test_imgs = normalize_img(test_imgs)
test_classes = np.asarray(test_classes)
print(test_imgs.shape)
print(test_classes.shape)

# evaluate the model for testing
predictions = sess.run(tf.argmax(logits,1), feed_dict={x: batch_x, y: batch_y})
print(predictions)
#print('Accuracy on web images = {:.3f} '.format(web_test_accuracy))



# saving the model now
saver.save(sess, './model/traffic')
print("Model saved")