import os
import logging

from datetime import datetime
from collections import OrderedDict

# Disable TF warnings about speed up and future warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Disable warnings from h5py
import warnings
warnings.filterwarnings("ignore", category = FutureWarning)

# Audio processing and DL frameworks 

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import keras
from keras import backend as K
from keras.models import load_model

from tensorflow.python.lib.io import file_io

from .struct import *
from .model import *
import argparse
# Constants
song_samples = 660000
genres = {'metal': 0, 'disco': 1, 'classical': 2, 'hiphop': 3, 'jazz': 4, 
          'country': 5, 'pop': 6, 'blues': 7, 'reggae': 8, 'rock': 9}
num_genres = len(genres)

def main(args):

    exec_time = datetime.now().strftime('%Y%m%d%H%M%S')


    # Start

    # Check if the directory path to GTZAN files was inputed
    if not args.job_dir:
        raise ValueError("File path to model should be passed in test mode.")



    # Read the files to memory and split into train test
    X, y = read_data(args.job_dir, genres, song_samples)
    print("X shape",X.shape)
    print("y shape",y.shape)
    # Transform to a 3-channel image
    X_stack = np.squeeze(np.stack((X,) * 3, -1))
    print("X_stack shape",X_stack.shape)
    X_train, X_test, y_train, y_test = train_test_split(X_stack, y, test_size=0.3, random_state=42, stratify = y)
    print("split done")

    # Training step
    input_shape = X_train[0].shape
    print(input_shape)
    cnn = build_model(input_shape, num_genres)
    cnn.compile(loss=keras.losses.categorical_crossentropy,
            optimizer=keras.optimizers.Adam(),
            metrics=['accuracy'])

    hist = cnn.fit(X_train, y_train,
            batch_size = 128,
            epochs = 1,
            verbose = 1,
            validation_data = (X_test, y_test))

    # Evaluate
    score = cnn.evaluate(X_test, y_test, verbose = 0)
    print("val_loss = {:.3f} and val_acc = {:.3f}".format(score[0], score[1]))


    # Save the model
    cnn.save('model.h5')
    with file_io.FileIO('model.h5', mode='r') as input_f:
            with file_io.FileIO(args.job_dir + 'model/model_test{}.h5'.format(exec_time), mode='w+') as output_f:
                output_f.write(input_f.read())



if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Music Genre Recognition on GTZAN')

    # Required arguments
    #parser.add_argument('-t', '--type', help='train or test mode to execute', type=str, required=True)

    # Nearly optional arguments. Should be filled according to the option of the requireds
    parser.add_argument('--job-dir', help='Path to the root directory with GTZAN files',required=True)
    parser.add_argument('-m', '--model', help='If choosed test, path to trained model', type=str)
    parser.add_argument('-s', '--song', help='If choosed test, path to song to classify', type=str)
    args = parser.parse_args()


    # Call the main function
    main(args)