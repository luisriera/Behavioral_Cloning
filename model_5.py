#
# Author: Luis G Riera
# Objective: Training a Model then cloned to drive a car simulator
#
'''
    This program find the Driving line interpolating from the given image
'''
import csv
import cv2
import numpy as np
import os
import sklearn
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Convolution2D, \
    MaxPooling2D, Cropping2D, Dropout

# root_path = 'C:\\Users\\Luis\\Desktop\\windows_sim\\record_data\\runs\\'
root_path = 'C:\\Users\\Luis\\Desktop\\windows_sim\\data\\'
IMG_path = root_path + 'IMG\\'
os.chdir(root_path)

epochs = 2
batch = 32

samples = []
with open(root_path + 'driving_log-Copy.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader, None)  # skip the headers
    for line in reader:
        samples.append(line)

from sklearn.model_selection import train_test_split

train_samples, validation_samples = train_test_split(samples, test_size=0.2)

print('There are {} samples'.format(len(samples)))

# root_path = 'C:\\Users\\Luis\\Desktop\\windows_sim\\record_data\\runs\\'
root_path = 'C:\\Users\\Luis\\Desktop\\windows_sim\\data\\'
IMG_path = root_path + 'IMG\\'
os.chdir(root_path)

# Convert a given image into gray scale
def imageToGray(image_data):
    return np.reshape(cv2.cvtColor(image_data, cv2.COLOR_RGB2GRAY), (160, 320, 1))


def generator(samples, batch_size=batch):
    ''''
     Generator to load data and preprocess it on the fly, in batch size portions
     Taken from: Udacity SDCND - Term-1: Behavioral Cloning
    '''
    num_samples = len(samples)
    while 1:  # Loop forever so the generator never terminates
        # shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = IMG_path + batch_sample[0].split('\\')[-1]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])

                if center_image == None:
                    print("Invalid image path:", name)
                else:
                    images.append(imageToGray(center_image))
                    angle = float(center_angle)
                    angles.append(angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch)
validation_generator = generator(validation_samples, batch_size=batch)


#-------------------------#
# NVIDIA end-to-end model #
#-------------------------#

# Trimmed image format
row, col, ch = 160, 320, 1

model = Sequential()

# Normalizing Layer
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(row, col, ch)))
model.add(Cropping2D(cropping=((35, 5), (0, 0))))

# Convolution layers with Relu activation
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))

# Flatten layer
model.add(Flatten())

# Dense layers with Dropout
model.add(Dense(16896))
model.add(Dropout(0.2))
model.add(Dense(100))
model.add(Dropout(0.2))
model.add(Dense(50))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

model.summary()
# Fit the model
history_obj = model.fit_generator(train_generator,
                                  validation_data=validation_generator,
                                  samples_per_epoch=len(train_samples),
                                  nb_val_samples=len(validation_samples),
                                  nb_epoch=3,
                                  verbose=1
                                  )

model.save('model_5.h5')

### print the keys contained in the history object
print(history_obj.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_obj.history['loss'])
plt.plot(history_obj.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

exit()
