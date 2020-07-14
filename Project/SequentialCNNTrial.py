import pandas as pd
import time
import numpy as np
from keras_preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D

start = time.time()

#Convert train and test data to dataframes
train_dataframe = pd.read_csv("train.csv", dtype=str)
test_dataframe = pd.read_csv("test.csv", dtype=str)


print (test_dataframe['classID'].as_matrix())
#Renaming
for index, row in train_dataframe.iterrows():
    row["slice_file_name"] = row["slice_file_name"].replace(".wav", "")
    row["slice_file_name"] = row["slice_file_name"] + ".png"

for index, row in test_dataframe.iterrows():
    row["slice_file_name"] = row["slice_file_name"].replace(".wav", "")
    row["slice_file_name"] = row["slice_file_name"] + ".png"

#Keras image generator
image_data_generator = ImageDataGenerator(rescale=1./255., validation_split = 0.125)

train_generator = image_data_generator.flow_from_dataframe(
    dataframe = train_dataframe,
    directory = "F:\\ml_img\\image_train_valid\\",
    x_col = "slice_file_name",
    y_col = "class",
    subset = "training",
    batch_size = 64,
    seed = 51, #Random random
    shuffle = True, #Shuffling
    class_mode = "categorical",
    target_size = (32, 32))

validation_generator = image_data_generator.flow_from_dataframe(
    dataframe = train_dataframe,
    directory = "F:\\ml_img\\image_train_valid\\",
    x_col = "slice_file_name",
    y_col = "class",
    subset = "validation",
    batch_size = 64,
    seed = 51,
    shuffle = True,
    class_mode = "categorical",
    target_size = (32, 32))

model = Sequential() #Sequential 2D CNN
model.add(Conv2D(32, (3, 3), padding="same", input_shape=(32,32,3)))
model.add(Activation("relu")) #Linear activation function
model.add(Conv2D(64, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2))) #Max pooling to decrease the features
model.add(Dropout(0.25)) #Droping out the unnecessary features

model.add(Conv2D(64, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(Conv2D(64, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(128, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(Conv2D(128, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Flatten()) #Compressing the remaining features
model.add(Dense(512))
model.add(Activation("relu"))
model.add(Dropout(0.5))

model.add(Dense(10, activation="softmax")) #Classification function
model.compile(loss="categorical_crossentropy",metrics=["accuracy"], optimizer="adam")
model.summary()

STEP_SIZE_TRAIN = train_generator.n//train_generator.batch_size
STEP_SIZE_VALID = validation_generator.n//validation_generator.batch_size

model.fit_generator(generator=train_generator, #Fitting the generator to model
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=validation_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=15) #Gradient
model.evaluate_generator(generator=validation_generator, steps=STEP_SIZE_VALID)


test_datagen=ImageDataGenerator(rescale=1./255.)
test_generator=test_datagen.flow_from_dataframe(
    dataframe=test_dataframe,
    directory="F:\\ml_img\\images_test\\",
    x_col="slice_file_name",
    y_col=None,
    batch_size=64,
    seed=51,
    shuffle=False,
    class_mode=None,
    target_size=(32, 32))

STEP_SIZE_TEST=test_generator.n//test_generator.batch_size

test_generator.reset()
pred=model.predict_generator(test_generator,
steps=STEP_SIZE_TEST,
verbose=1)

predicted_class_indices = np.argmax(pred,axis=1)
print(predicted_class_indices)



i = 0
true_count = 0;
while i < len(predicted_class_indices):
    if (int((test_dataframe['classID'].as_matrix())[i]) == predicted_class_indices[i]):
        true_count += 1
    i += 1

print(true_count / len(predicted_class_indices))
end = time.time()
print("Total time: ", end - start, " seconds")