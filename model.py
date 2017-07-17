#   Author: Luis G.Riera
#   date: 15-Jul-2017
#
#   Objective:
#   This program use as input images with their respective steering angles
#   to train a model which it is later feed to a driving simulator.
#

import os
import csv
import cv2
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Convolution2D, Cropping2D, Dropout

root_path = 'C:\\Users\\Luis\\Desktop\\windows_sim\\data\\'
IMG_path = root_path + 'IMG\\'
os.chdir(root_path)

# Model input parameters
epochs, batch = 4, 32

# Image input format
row, col, ch = 160, 320, 3

samples = []
with open(root_path + 'driving_log-Flips.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader, None)  # skip the headers
    for line in reader:
        samples.append(line)

train_samples, validation_samples = train_test_split(samples, test_size=0.2)

print('There are a total {} samples images'.format(len(samples)))


def generator(sample_images, batch_size=batch):
    # Generator: load data and preprocesses input data on the fly, in batch size portions
    # Taken from: Udacity SDCND - Term-1: Behavioral Cloning

    num_samples = len(sample_images)
    while True:  # Loop forever so the generator never terminates
        for offset in range(0, num_samples, batch_size):
            batch_samples = sample_images[offset:offset + batch_size]

            images = []  # Place holder for the images
            angles = []  # Place holder for the steering angles
            for batch_sample in batch_samples:
                name = IMG_path + batch_sample[0].split('\\')[-1]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])

                # Ensure that name the image in the CVS exist
                if center_image is None:
                    print("Invalid image path:", name)
                else:
                    images.append(center_image)
                    angle = float(center_angle)
                    angles.append(angle)

            x_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(x_train, y_train)


# ---------------------------------------------------#
# NVIDIA end-to-end model with added Dropout layers #
# ---------------------------------------------------#
model = Sequential()
# Normalization Layer
model.add(Cropping2D(cropping=((70, 25), (1, 1)), input_shape=(160, 320, 3)))  # Trim image to only see the road
model.add(Lambda(lambda x: x / 255.0 - 0.5))  # Normalize colours

# Convolution layers with Relu activation
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))

# Flatten layer
model.add(Flatten())

# Dense layers with Dropout
model.add(Dense(1300))
model.add(Dropout(0.2))
model.add(Dense(100))
#model.add(Dropout(0.1))
model.add(Dense(50))
model.add(Dense(1))

# compile and train the model using the generator function
# Using "Means Square Root" for losses calculation and Adam for optimizer
model.compile(loss='mse', optimizer='adam')

# Print model summary table
model.summary()

# Fit the model
train_generator = generator(train_samples, batch_size=batch)
validation_generator = generator(validation_samples, batch_size=batch)
history_obj = model.fit_generator(train_generator,
                                  nb_epoch=epochs,
                                  validation_data=validation_generator,
                                  samples_per_epoch=len(train_samples),
                                  nb_val_samples=len(validation_samples),
                                  verbose=1)

model.save('model.h5')

# print the keys contained in the history object
print(history_obj.history.keys())

# plot the training and validation loss for each epoch
plt.plot(history_obj.history['loss'])
plt.plot(history_obj.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

exit()
