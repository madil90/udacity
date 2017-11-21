import pickle
import tensorflow as tf
# TODO: import Keras layers you need here

from keras.layers import Input, Dense, Flatten
from keras.models import Model

flags = tf.app.flags
FLAGS = flags.FLAGS

# command line flags
flags.DEFINE_string('training_file', 'vgg_cifar10_100_bottleneck_features_train.p', "Bottleneck features training file (.p)")
flags.DEFINE_string('validation_file', 'vgg_cifar10_bottleneck_features_validation.p', "Bottleneck features validation file (.p)")

flags.DEFINE_string('batch_size', 32, "Batch size for training")
flags.DEFINE_string('no_epochs', 10, "No of epochs for training")




def load_bottleneck_data(training_file, validation_file):
    """
    Utility function to load bottleneck features.

    Arguments:
        training_file - String
        validation_file - String
    """
    print("Training file", training_file)
    print("Validation file", validation_file)

    with open(training_file, 'rb') as f:
        train_data = pickle.load(f)
    with open(validation_file, 'rb') as f:
        validation_data = pickle.load(f)

    X_train = train_data['features']
    y_train = train_data['labels']
    X_val = validation_data['features']
    y_val = validation_data['labels']

    return X_train, y_train, X_val, y_val


def main(_):
    # load bottleneck data
    X_train, y_train, X_val, y_val = load_bottleneck_data(FLAGS.training_file, FLAGS.validation_file)

    print(X_train.shape, y_train.shape)
    print(X_val.shape, y_val.shape)

    # TODO: define your model and hyperparams here
    # make sure to adjust the number of classes based on
    # the dataset
    # 10 for cifar10
    # 43 for traffic
    n_classes = 43 # for cifar 10
    inputs = Input(shape=X_train.shape[1:])
    x = Flatten()(inputs)
    x = Dense(n_classes, activation='softmax')(x)
    model = Model(inputs, x)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # train model
    model.fit(X_train, y_train, epochs=FLAGS.no_epochs, batch_size=FLAGS.batch_size, validation_data=(X_val, y_val), shuffle=True)

    # TODO: train your model here
    print('What? ')


# parses flags and calls the `main` function above
if __name__ == '__main__':
    tf.app.run()
