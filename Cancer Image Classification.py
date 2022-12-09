# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 14:41:21 2022

@author: BD
"""

import numpy as np
import tensorflow as tf      #!!! clean up this import, only used once
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, BatchNormalization, Conv2D, MaxPool2D, Rescaling
#from keras.layers.advanced_activations import PReLU
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.metrics import CategoricalCrossentropy, CategoricalAccuracy, AUC
from tensorflow_addons.metrics import F1Score
from tensorflow.keras.callbacks import ModelCheckpoint, History
from os.path import normpath
from sklearn.model_selection import KFold, ParameterGrid
from tensorflow.keras.preprocessing.image import ImageDataGenerator

image_size = (256,256)
batch_size = 16
train_path = r'C:/Users/yblad/Documents/For Bsc/Year 3/AI/Assessed Work/Project/Code/Original_train'
test_path = r'C:/Users/yblad/Documents/For Bsc/Year 3/AI/Assessed Work/Project/Code/Original_test'
drop_out = 0.2

print("Loading training data:")    
#ds_tr = keras.preprocessing.image_dataset_from_directory(
#    train_path,
#    validation_split=0.2,
#    subset="training",
#    seed=208,
#    image_size=image_size,
#    batch_size=1,
#    label_mode='categorical'
#)

ds_tr = ImageDataGenerator(validation_split=0.2, preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=train_path, batch_size=1,
                         seed=208, subset='training')
 
print("Loading validation data:")   

ds_val = ImageDataGenerator(validation_split=0.2, preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=train_path, batch_size=1,
                         seed=208, subset='validation')
 
#ds_val = keras.preprocessing.image_dataset_from_directory(
#    train_path,
#    validation_split=0.2,
#    subset="validation",
#    seed=208,
#    image_size=image_size,
#    batch_size=1,
#    label_mode='categorical'
#)
print("Loading test data:") 
ds_test = keras.preprocessing.image_dataset_from_directory(
        test_path,
        image_size=image_size,
        label_mode='categorical',
       batch_size=1,
    )

#class_names = ds_tr.class_names   
#print(f"\n Class names are {class_names}")

print(ds_tr)

tr_length = len(ds_tr)
val_length = len(ds_val)

history = History()
checkpoint_path = r"C:\Users\yblad\Documents\For Bsc\Year 3\AI\Assessed Work\Project\Code\Checkpoints\1.ckpt"
checkpoint_path = normpath(checkpoint_path)

cp_callback = ModelCheckpoint(
    filepath=checkpoint_path, 
    verbose=1, 
    save_weights_only=True,
    save_freq= 5 * tr_length // batch_size)

#Preprocessing, !!! there is a bug I need to fix here
#ds_tr = keras.applications.vgg16.preprocess_input(ds_tr)
#ds_val = keras.applications.vgg16.preprocess_input(ds_val)
#ds_test = keras.applications.vgg16.preprocess_input(ds_test)

#Augmentation
Flip = keras.layers.RandomFlip()
Rotate = keras.layers.RandomRotation(0.25)

def augment(image, label):
    image = Flip(image)
    image = Rotate(image)
    
    return image, label
    
  
AUTOTUNE = tf.data.AUTOTUNE

#Augmentation broken currently, see bookmarks to work out fix
ds_tr = ds_tr.map(augment, num_parallel_calls=4)

ds_tr = ds_tr.cache().prefetch(buffer_size=AUTOTUNE)
ds_val = ds_val.cache().prefetch(buffer_size=AUTOTUNE)

#Batch the data
ds_tr = ds_tr.unbatch()
ds_tr = ds_tr.batch(batch_size=batch_size)
ds_val = ds_val.unbatch()
ds_val = ds_val.batch(batch_size=batch_size)

#Really quick model just to test things are working
model1 = Sequential([
        Rescaling(1./255),
        Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding = 'same', input_shape=(224,224,3)),
        Dropout(drop_out),
        MaxPool2D(pool_size=(2, 2), strides=2),
        Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding = 'same'),
        Dropout(drop_out),
        MaxPool2D(pool_size=(2, 2), strides=2),
        Flatten(),
        Dense(units=4, activation='softmax'),
])

#model1.compile(optimizer=Adam(learning_rate=0.0001),
#              loss='categorical_crossentropy',
#              metrics=[F1Score(num_classes=4, average='weighted'),
#                       AUC(curve='PR'),CategoricalAccuracy()]
#)


#model1.fit(x = ds_tr,
#          validation_data=ds_val,
#          epochs=50,
#          callbacks=[cp_callback, history],
#          verbose=2
#)

#!!! Testing of cross-val. Will drop seperate validation
#data once using this.
#Adapted from code by Uche Onyekpe
kf = KFold(n_splits=5, shuffle=True, random_state=247)
param_combination = 1
i = 0

#paramater grid to search
params= {'optimizer':[Adam()]}

#make into iterable
params = ParameterGrid(params)

models = []
models.append(model1)

for train_index, test_index in kf.split(ds_tr):
    for model in models:
        #Get compiler options from params
        if 'optimizer' in params[i]:
            optimizer = params[i]['optimizer']
            if optimizer == 'Adam':
                optimizer = Adam()
            
        model.compile(optimizer=optimizer)


#print(model.summary())

#To try diff batch size
#ds = ds.unbatch()
#ds = ds.batch(batch_size=n)

