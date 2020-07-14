import pandas as pd
import os
import struct
import librosa
import csv
import librosa.display
import numpy as np
import matplotlib.pyplot as pyplot
import soundfile as sf
from glob import glob
import keras
from keras_preprocessing.image import ImageDataGenerator
from keras import layers
from keras import models
from keras.layers.advanced_activations import LeakyReLU
import keras.backend as K
from keras.optimizers import Adam
from keras.layers import Conv2D, MaxPooling2D, Input, Dense, Flatten
from keras.models import Model
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split


""""
with open( "UrbanSound8K.csv", 'r') as dataset_info:
    lines = csv.reader(dataset_info)
    for row in lines:
        sound_name = ""
        if(row[5] == "1" or row[5] == "2"):
            sound_name = row[0].replace('.wav', '')
            sound_file_temp, sample_rate = sf.read("fold" + row[5] + "/" +sound_name + ".wav",dtype="float32")
            sound_file_temp = sound_file_temp.T
            sound_file_temp2 = librosa.resample(sound_file_temp, sample_rate, 11025/2)
            sound_file = librosa.to_mono(sound_file_temp2)
            #sound_file, sample_rate = librosa.load("fold" + row[5] + "/" +sound_name + ".wav", sr=None)
            fg = pyplot.figure(figsize=[1, 1])
            spectogram = librosa.feature.melspectrogram(y=sound_file, sr=sample_rate)
            librosa.display.specshow(librosa.power_to_db(spectogram, ref=np.max))
            filename = "images" + row[5] + "/" +sound_name + '.png'
            pyplot.savefig(filename, dpi=500, bbox_inches='tight',pad_inches=0)
            pyplot.close()
            fg.clf()
            pyplot.close(fg)
            pyplot.close('all')

            print("fold" + row[5] + "/" +sound_name + ".wav")

"""

train_dataframe = pd.read_csv('train.csv', dtype=str)
test_dataframe = pd.read_csv('test.csv', dtype=str)

for index, row in train_dataframe.iterrows():
    row["slice_file_name"] = row["slice_file_name"].replace('.wav', '')
    row["slice_file_name"] = row["slice_file_name"] + '.png'

for index, row in test_dataframe.iterrows():
    row["slice_file_name"] = row["slice_file_name"].replace('.wav', '')
    row["slice_file_name"] = row["slice_file_name"] + '.png'

image_data_generator = ImageDataGenerator(rescale=1./255.,validation_split=0.125)


train_generator=image_data_generator.flow_from_dataframe(
    dataframe = train_dataframe,
    directory = "images1/",
    x_col = "slice_file_name",
    y_col = "class",
    subset = "training",
    batch_size = 32,
    seed = 42,
    shuffle = True,
    class_mode = "categorical",
    target_size = (64, 64))

validation_generator = image_data_generator.flow_from_dataframe(
    dataframe = train_dataframe,
    directory = "images1/", #bok
    x_col = "slice_file_name", #ok
    y_col = "class", #ok
    subset = "validation", #ok
    batch_size = 32,
    seed = 42,
    shuffle = True, #ok
    class_mode = "categorical", #ok
    target_size = (64,64))

model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=(64,64,3)))
model.add(Activation('linear'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('linear'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('linear'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('linear'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.5))
model.add(Conv2D(128, (3, 3), padding='same'))
model.add(Activation('linear'))
model.add(Conv2D(128, (3, 3)))
model.add(Activation('linear'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))


model.add(Flatten())
model.add(Dense(512))
model.add(Activation('linear'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
model.compile(loss="categorical_crossentropy",metrics=["accuracy"], optimizer='adam')
model.summary()

""" Non Sequential Trial
# First, define the vision modules
num_rows = 40
num_columns = 174
num_channels = 1
    X = np.array(train_dataframe.feature.tolist())
y = np.array(train_dataframe.class_label.tolist())
le = LabelEncoder()
yy = to_categorical(le.fit_transform(y))
x_train, x_test, y_train, y_test = train_test_split(X, yy, test_size=0.2, random_state = 42)
x_train = x_train.reshape(x_train.shape[0], num_rows, num_columns, num_channels)
x_test = x_test.reshape(x_test.shape[0], num_rows, num_columns, num_channels)

num_labels = yy.shape[1]
filter_size = 2
digit_input = Input(shape=(27, 27, 1))
x = Conv2D(64, (3, 3))(digit_input)
x = Conv2D(64, (3, 3))(x)
x = MaxPooling2D((2, 2))(x)
out = Flatten()(x)

vision_model = Model(digit_input, out)

# Then define the tell-digits-apart model
digit_a = Input(shape=(27, 27, 1))
digit_b = Input(shape=(27, 27, 1))

# The vision model will be shared, weights and all
out_a = vision_model(digit_a)
out_b = vision_model(digit_b)

concatenated = keras.layers.concatenate([out_a, out_b])
out1 = Dense(1, activation='sigmoid')(concatenated)
out2 = Dense(1, activation='sigmoid')(concatenated)

classification_model = Model([digit_a, digit_b], outputs=[out1, out2])
classification_model.compile(loss="categorical_crossentropy",metrics=["accuracy"], optimizer='adam')
classification_model.summary()
#Fitting keras model, no test gen for now
"""
STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=validation_generator.n//validation_generator.batch_size
#STEP_SIZE_TEST=test_generator.n//test_generator.batch_size

model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=validation_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=20
)
model.evaluate_generator(generator=validation_generator, steps=STEP_SIZE_VALID
)
""" Non Sequential Fit
model.fit(X,Y,
          batch_size=32,nb_epoch=20,
          validation_data=(np.array(X)),
          #callbacks=[early_stop])
)
"""