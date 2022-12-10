# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 14:41:21 2022

@author: BD
"""

import numpy as np
import tensorflow as tf
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
import matplotlib.pyplot as plt
from math import ceil


image_size = (256,256)
batch_size = 16
train_path = r'C:/Users/yblad/Documents/For Bsc/Year 3/AI/Assessed Work/Project/Code/Original_train'
test_path = r'C:/Users/yblad/Documents/For Bsc/Year 3/AI/Assessed Work/Project/Code/Original_test'
drop_out = 0.2

print("Loading training data:")    
ds_tr = keras.preprocessing.image_dataset_from_directory(
    train_path,
    validation_split=0.2,
    subset="training",
    seed=208,
    image_size=image_size,
    batch_size=1,
    label_mode='categorical',
    shuffle=True
)

print("Loading validation data:")   
 
ds_val = keras.preprocessing.image_dataset_from_directory(
    train_path,
    validation_split=0.2,
    subset="validation",
    seed=208,
    image_size=image_size,
    batch_size=1,
    label_mode='categorical',
    shuffle=True
)

print("Loading test data:") 
ds_test = keras.preprocessing.image_dataset_from_directory(
        test_path,
        image_size=image_size,
        label_mode='categorical',
       batch_size=1,
       shuffle=True
    )

class_names = ds_tr.class_names   
print(f"\n Class names are {class_names}")

#Preprocessing. We pass just the image tensor portion
#of the training data to the adapt method of a normalise layer.
#This gives us a normaliser which does (input - mean) / sqrt(var)
#based on the mean and var from the training data channels only

features = ds_tr.map(lambda x, y: x)
norm_layer = tf.keras.layers.Normalization(axis=None)
norm_layer.adapt(features)

def preprocessing(image, label):
    #image = keras.applications.vgg16.preprocess_input(image)
    image = norm_layer(image)
    
    return image, label

ds_tr = ds_tr.map(preprocessing, num_parallel_calls=4)
ds_val = ds_val.map(preprocessing, num_parallel_calls=4)
ds_test = ds_test.map(preprocessing, num_parallel_calls=4)

#Show images, adapted from code by Uche Onyekpe
def plotImages(images):
    fig, axes = plt.subplots(1, 10, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip(images, axes):
        ax.imshow(img[0,:,:,:])
        ax.axis('off')
    plt.tight_layout()
    plt.show()

features = ds_tr.map(lambda x, y: x)

plotImages(features)

print(ds_tr)

tr_length = len(ds_tr)
val_length = len(ds_val)

#Augmentation
Flip = keras.layers.RandomFlip()
Rotate = keras.layers.RandomRotation(0.25)

def augment(image, label):
    image = Flip(image)
    image = Rotate(image)
    
    return image, label
    
  
AUTOTUNE = tf.data.AUTOTUNE

ds_tr = ds_tr.map(augment, num_parallel_calls=4)

ds_tr = ds_tr.cache().prefetch(buffer_size=AUTOTUNE)
ds_val = ds_val.cache().prefetch(buffer_size=AUTOTUNE)

#Batch the data
ds_tr = ds_tr.unbatch()
#ds_tr = ds_tr.batch(batch_size=batch_size)
ds_val = ds_val.unbatch()
#ds_val = ds_val.batch(batch_size=batch_size)

#Really quick model just to test things are working
model = Sequential([
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

checkpoint_path = r"C:\Users\yblad\Documents\For Bsc\Year 3\AI\Assessed Work\Project\Code\Checkpoints\1.ckpt"
checkpoint_path = normpath(checkpoint_path)

cp_callback = ModelCheckpoint(
    filepath=checkpoint_path, 
    verbose=1, 
    save_weights_only=True,
    save_freq= 5 * tr_length // batch_size)

history = History()

model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=[F1Score(num_classes=4, average='weighted'),
                       AUC(curve='PR'),CategoricalAccuracy()]
)

#model.fit(x = ds_tr,
#          validation_data=ds_val,
#          epochs=50,
#          callbacks=[cp_callback, history],
#          verbose=2
#)

params = {'optimizer':['Adam']}
params = ParameterGrid(params)

cv_split = 5

models =[]
models.append(model)

#for train_index, test_index in kf.split(ds_tr):
    
def cv(cv_split, train_data, tr_length, models):
    
    #train_data = tf.Variable(train_data)
    fold_fraction = 1/cv_split
    train_fraction= 1 - fold_fraction
    
    #val_size = val_fraction * len(train_data)
    train_size = train_fraction * tr_length
    train_fold_size = fold_fraction * tr_length
    
    #Round the fold size and train size up. As we are using tf.window()
    #this will ensure all data is placed in a fold. Will preference an additional
    #sample into training when tr_length mod cv_split neq 0.
    
    train_size = ceil(train_size)
    train_fold_size = ceil(train_fold_size)
    
    train_folds = []
    
    #Randomis date. We can maintain a smaller buffer as data is already
    #shuffled on load. We just want to make sure each iteration sees
    #a different order of the data
    train_data = train_data.shuffle(tr_length//4, reshuffle_each_iteration=True)
    
    for model in models:
        #Split data.

        train_data = train_data.take(train_size)
        val_fold = train_data.skip(train_size)
    
        train_folds = train_data.window(train_fold_size, stride = 1, shift = train_fold_size,
                                        drop_remainder=False)
        
        
        #get batch_size from params
        for i in range(len(params)):
            if 'batch_size' in params[i]:
                batch_size = params[i]['batch_size']
            else:
                batch_size = 16
                    
            #Get compiler options from params
            if 'optimizer' in params[i]:
                optimizer = params[i]['optimizer']
                if optimizer == 'Adam':
                    optimizer = Adam()
                
        features = train_folds.flat_map(lambda x, y: x.batch(batch_size))
        labels = train_folds.flat_map(lambda x, y: y.batch(batch_size))
        train_data = tf.data.Dataset.from_tensor_slices((features, labels))
        #train_data = train_folds.flat_map(lambda x, y: tf.data.Dataset.from_tensor_slices((
        #    x.batch(batch_size), y.batch(batch_size))))

        print(train_data)
        model.compile(optimizer=optimizer)
        
cv(5, ds_tr, tr_length, models)

#print(model.summary())

#To try diff batch size
#ds = ds.unbatch()
#ds = ds.batch(batch_size=n)

