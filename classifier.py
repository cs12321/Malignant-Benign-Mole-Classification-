#Importing the Keras libraries and packages
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense

#Initializing the CNN
classifier = Sequential()

#Step 1-Convolution
classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))

#Step 2-Pooling
classifier.add(MaxPooling2D(pool_size = (2,2)))

#Step 3-Flattening
classifier.add(Flatten())

#Step 4-Fully Connected Layer
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

#Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#Part 2-Fitting the CNN to the Images
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'skin_cancer/train',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')


test_set = test_datagen.flow_from_directory(
        'skin_cancer/test',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(
        training_set,
        steps_per_epoch=None,
        epochs=10,
        validation_data=test_set,
        validation_steps=None)

from keras.preprocessing import image
import numpy as np

#Classify an individual image

test_image = image.load_img('skin_cancer/malignant_mole.jpg', target_size = ( 64, 64))

test_image = image.img_to_array(test_image)

test_image = np.expand_dims(test_image, axis = 0)

result = classifier.predict_classes(test_image)

print(result)

training_set.class_indices

if result[0][0] == 1:
    prediction = 'malignant'
    print("The image is of a ",prediction)

else:
    prediction = 'benign'
    print("The image is of a ",prediction) 
