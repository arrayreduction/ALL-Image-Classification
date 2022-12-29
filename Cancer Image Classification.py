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
from tensorflow.keras.optimizers import Adam, Adagrad
from tensorflow.keras.metrics import CategoricalCrossentropy, CategoricalAccuracy, AUC
from tensorflow_addons.metrics import F1Score
from tensorflow.keras.callbacks import ModelCheckpoint, History
from os.path import normpath
from sklearn.model_selection import KFold, ParameterGrid
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from math import ceil
import gc

physical_devices = tf.config.list_physical_devices('GPU')
try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
  # Invalid device or cannot modify virtual devices once initialized.
  pass

with tf.device("/cpu:0"):

    image_size = (224,224)
    train_path = r'C:/Users/yblad/Documents/For Bsc/Year 3/AI/Assessed Work/Project/Code/Original_train'
    test_path = r'C:/Users/yblad/Documents/For Bsc/Year 3/AI/Assessed Work/Project/Code/Original_test'
    
    print("Loading training data:")    
    ds_tr = keras.preprocessing.image_dataset_from_directory(
        train_path,
        validation_split=0.2,
        subset='training',
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
        subset='validation',
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
    
    #Preprocessing. We pass just the image tensor portion
    #of the training data to the adapt method of a normalise layer.
    #This gives us a normaliser which does (input - mean) / sqrt(var)
    #based on the mean and var from the training data channels only
    
    crop_layer = tf.keras.layers.Cropping2D(cropping=((0, 30), (0, 0)))
    
    def preprocessing_1(image, label):
        image = crop_layer(image)
        
        return image, label
    
    def preprocessing_2(image, label):
        image = norm_layer(image)
        
        return image, label
    
    ds_tr = ds_tr.map(preprocessing_1, num_parallel_calls=4)
    ds_val = ds_val.map(preprocessing_1, num_parallel_calls=4)
    ds_test = ds_test.map(preprocessing_1, num_parallel_calls=4)
    
    ds_tr = ds_tr.map(augment, num_parallel_calls=4)
    
    features = ds_tr.map(lambda x, y: x)
    norm_layer = tf.keras.layers.Normalization(axis=None)
    norm_layer.adapt(features)
    
    ds_tr = ds_tr.map(preprocessing_2, num_parallel_calls=4)
    ds_val = ds_val.map(preprocessing_2, num_parallel_calls=4)
    ds_test = ds_test.map(preprocessing_2, num_parallel_calls=4)
    
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
    
    
    #checkpoint_path = r"C:\Users\yblad\Documents\For Bsc\Year 3\AI\Assessed Work\Project\Code\Checkpoints\1.ckpt"
    #checkpoint_path = normpath(checkpoint_path)
    
    #cp_callback = ModelCheckpoint(
    #    filepath=checkpoint_path, 
    #    verbose=1, 
    #    save_weights_only=True,
    #    save_freq= 5 * tr_length // batch_size)
    
    #history = History()
    
    #model.compile(optimizer=Adam(learning_rate=0.0001),
    #              loss='categorical_crossentropy',
    #              metrics=[F1Score(num_classes=4, average='macro'),
    #                       AUC(curve='PR'),CategoricalAccuracy()]
    #)
    
    #model.fit(x = ds_tr,
    #          validation_data=ds_val,
    #          epochs=50,
    #          callbacks=[cp_callback, history],
    #          verbose=2
    #)
      
def get_param_vars(params):
    '''Unbundles dictionary items into variables'''
    #get batch_size from params
    if 'batch_size' in params:
        batch_size = params['batch_size']
    else:
        print("WARN: Using default batch size of 16")
        batch_size = 16
            
    #Get compiler options from params
    if 'learning_rate' in params:
        learning_rate = params['learning_rate']
    else:
        print("WARN: Using default learning rate of 0.0001")
        learning_rate = 0.0001
        
    if 'optimizer' in params:
        optimizer = params['optimizer']
        if optimizer == 'Adam':
            optimizer = Adam(learning_rate)
        elif optimizer == 'Adagrad':
            optimizer = Adagrad(learning_rate)
    else:
        print("WARN: Using default optimiser; Adam")
        optimizer = Adam(learning_rate)
            
    #Get fit() options from params
    if 'epochs' in params:
        epochs = params['epochs']
    else:
        print("WARN: Using default epochs of 50")
        epochs = 50   
        
    return batch_size, optimizer, epochs

#def get_f1(preds, data):
#        metric = F1Score(num_classes=4, average='macro')
#
#        _, tr_labels = tuple(zip(*data.unbatch()))
#        tr_labels = np.array(tr_labels)
#        print(tr_labels)
#        metric.update_state(tr_labels,preds)
#        
#        result = metric.result().numpy()
#        metric.reset_state()
#        
#        return result
    

def cv(cv_split, train_data, tr_length, model, param_grid, return_best=True):
    '''Cross validator. Designed to crossvalidate different paramaters on
    a single model arcitecture.'''
  
    #First run flag
    first_fold = True
        
    
    fold_fraction = 1/cv_split
    train_fraction= 1 - fold_fraction
    
    #val_size = val_fraction * len(train_data)
    train_size = train_fraction * tr_length
    train_fold_size = fold_fraction * tr_length
    
    #Round the fold size and train size up. As we are using tf.window()
    #this will ensure all data are placed in a fold.
    
    train_size = ceil(train_size)
    train_fold_size = ceil(train_fold_size)
    
    cv_scores = []
    
    for params in param_grid:
        metric_ave_tr = 0
        metric_ave_val = 0
        metric_scores_train = []
        metric_scores_val = []
        train_data = train_data.unbatch()
        
        #Randomise data order. We can maintain a smaller buffer as data are already
        #shuffled on load. We just want to make sure each iteration sees
        #a different order of the data
        train_data = train_data.shuffle(tr_length//4, reshuffle_each_iteration=False)
        
        batch_size, optimizer, epochs = get_param_vars(params)
        
        train_data = train_data.batch(train_fold_size)
        folds=[]
        
        #Put together list of folds
        
        for i in range(cv_split):
            folds.append(train_data.skip(i).take(1))
            
        assert(len(folds) == cv_split)
            
        #Put folds back into a proper dataset and train
        #On each i loop we pick a different fold to be the val_fold
        #We then add training folds until all folds are assigned
         
        for i in range(cv_split):
            train_folds = None
            
            for j in range(cv_split):
                if j == i:
                    val_fold = folds[j]
                else:
                    if train_folds is None:
                        train_folds = folds[j]
                    else:
                        train_folds = train_folds.concatenate(folds[j])
            
            
            print(f"Training fold {i}")
            
            train_folds = train_folds.unbatch()
            train_folds = train_folds.batch(batch_size)
            
            val_fold = val_fold.unbatch()
            val_fold = val_fold.batch(batch_size)
            
            #If this is the first fold compile a new model.
            #Otherwise, use clone_model() to make
            #a new copy of the model with re-initialised wieghts
            
            if first_fold:
                model.compile(optimizer=optimizer,
                              loss='categorical_crossentropy',
                              metrics=[F1Score(num_classes=4, average='macro'),
                                       CategoricalAccuracy()]
                )
            else:
                model = keras.models.clone_model(model)
                gc.collect()
                model.compile(optimizer=optimizer,
                              loss='categorical_crossentropy',
                              metrics=[F1Score(num_classes=4, average='macro'),
                                       CategoricalAccuracy()]
                )
                
            
            train_folds = train_folds.cache().prefetch(buffer_size=AUTOTUNE)
            #val_fold = val_fold.cache().prefetch(buffer_size=AUTOTUNE)
            
            model.fit(x = train_folds,
                      epochs=epochs,
                      verbose=2
            )
            
            f1 = model.evaluate(train_folds)[1]
            metric_scores_train.append(f1)      
            
            f1 = model.evaluate(val_fold)[1]
            metric_scores_val.append(f1)
            
            first_fold = False


        metric_ave_tr = np.mean(metric_scores_train)
        metric_ave_val = np.mean(metric_scores_val)
        cv_scores.append((metric_ave_tr, metric_ave_val, params))
    
    if return_best:
        tr_score, val_score, _ = zip(*cv_scores)
        idx = np.argmax(val_score)
        best = cv_scores[idx]
        
        return best
    else:
        return cv_scores

#VGG-like model.
drop_out = 0.35

model_vgg7 = Sequential([
        Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding = 'same',
               kernel_initializer='he_normal', input_shape=(194,224,3)),
        Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding = 'same',
               kernel_initializer='he_normal'),
        MaxPool2D(pool_size=(2, 2), strides=2),
        Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding = 'same',
               kernel_initializer='he_normal'),
        Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding = 'same',
               kernel_initializer='he_normal'),
        MaxPool2D(pool_size=(2, 2), strides=2),
        Flatten(),
        BatchNormalization(),
        Dense(units=12, activation='relu', kernel_initializer='he_normal'),
        BatchNormalization(),
        #Dropout(drop_out),
        Dense(units=12, activation='relu', kernel_initializer='he_normal'),
        BatchNormalization(),
        Dropout(drop_out),
        Dense(units=4, activation='softmax', kernel_initializer='he_normal'),
])

history = History()

params = {'optimizer':['Adam'],
          'epochs':[25],
          'batch_size':[8, 16],
          'learning_rate':[0.001,0.00001]}
params = ParameterGrid(params)
            
scores = cv(5, ds_tr, tr_length, model_vgg7, params)
print(f'\n Best score (val):\n (tr, val, params) \n {scores} \n')

best_params = scores[2]
batch_size, optimizer, epochs = get_param_vars(best_params)

ds_tr = ds_tr.unbatch()
ds_tr = ds_tr.batch(batch_size)

ds_val = ds_val.unbatch()
ds_val = ds_val.batch(batch_size)

model_vgg7 = keras.models.clone_model(model_vgg7)
model_vgg7.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=[F1Score(num_classes=4, average='macro'),
                       CategoricalAccuracy()]
              )

model_vgg7.fit(x = ds_tr,
          epochs=epochs,
          validation_data=ds_val,
          verbose=2,
          callbacks=[history]
)

print(model_vgg7.summary())

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validation')
plt.title("Vgg7 Loss (no L2 regularization)")
plt.legend()
plt.show()

plt.plot(history.history['f1_score'], label='train')
plt.plot(history.history['val_f1_score'], label='validation')
plt.title("Vgg7 F1 Score (no L2 regularization)")
plt.legend()
plt.show()

model_vgg7.evaluate(ds_tr)
model_vgg7.evaluate(ds_val)

model_vgg7.save_weights('Models/vgg7/vgg7_model_weights.pb')

model_vgg7_l2 = Sequential([
        Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding = 'same',
               kernel_initializer='he_normal', input_shape=(194,224,3), kernel_regularizer='l2'),
        Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding = 'same',
               kernel_initializer='he_normal', kernel_regularizer='l2'),
        MaxPool2D(pool_size=(2, 2), strides=2),
        Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding = 'same',
               kernel_initializer='he_normal', kernel_regularizer='l2'),
        Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding = 'same',
               kernel_initializer='he_normal', kernel_regularizer='l2'),
        MaxPool2D(pool_size=(2, 2), strides=2),
        Flatten(),
        BatchNormalization(),
        Dense(units=12, activation='relu', kernel_initializer='he_normal',
              kernel_regularizer='l2'),
        BatchNormalization(),
        #Dropout(drop_out),
        Dense(units=12, activation='relu', kernel_initializer='he_normal',
              kernel_regularizer='l2'),
        BatchNormalization(),
        Dropout(drop_out),
        Dense(units=4, activation='softmax', kernel_initializer='he_normal',
              kernel_regularizer='l2'),
])

params = {'optimizer':['Adam'],
          'epochs':[25],
          'batch_size':[16],
          'learning_rate':[0.001]}
params = ParameterGrid(params)

scores = cv(5, ds_tr, tr_length, model_vgg7_l2, params)
print(f'\n Best score (val):\n (tr, val, params) \n {scores} \n')

best_params = scores[2]
batch_size, optimizer, epochs = get_param_vars(best_params)

ds_tr = ds_tr.unbatch()
ds_tr = ds_tr.batch(batch_size)

ds_val = ds_val.unbatch()
ds_val = ds_val.batch(batch_size)

model_vgg7_l2 = keras.models.clone_model(model_vgg7_l2)
model_vgg7_l2.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=[F1Score(num_classes=4, average='macro'),
                       CategoricalAccuracy()]
              )

model_vgg7_l2.fit(x = ds_tr,
          epochs=epochs,
          validation_data=ds_val,
          verbose=2,
          callbacks=[history]
)

print(model_vgg7_l2.summary())

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validation')
plt.title("Vgg7 Loss (L2 regularization)")
plt.legend()
plt.show()

plt.plot(history.history['f1_score'], label='train')
plt.plot(history.history['val_f1_score'], label='validation')
plt.title("Vgg7 F1 Score (L2 regularization)")
plt.legend()
plt.show()

model_vgg7_l2.evaluate(ds_tr)
model_vgg7_l2.evaluate(ds_val)

#For running against test set, with final models only
#f1 = model_xxx.evaluate(ds_test)