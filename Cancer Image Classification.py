# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 14:41:21 2022

@author: BD
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, BatchNormalization, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam, Adagrad
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow_addons.metrics import F1Score
from tensorflow.keras.callbacks import History
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from math import ceil
import gc

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
    
    #Round the fold size and train size up. As we are using take/skip
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
                
            AUTOTUNE = tf.data.AUTOTUNE
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

def model_1(ds_tr, ds_val, tr_length):
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
    plt.title("Vgg7 Loss (drop out=0.35)")
    plt.legend()
    plt.show()
    
    plt.plot(history.history['f1_score'], label='train')
    plt.plot(history.history['val_f1_score'], label='validation')
    plt.title("Vgg7 F1 Score (drop out=0.35)")
    plt.legend()
    plt.show()
    
    model_vgg7.evaluate(ds_tr)
    model_vgg7.evaluate(ds_val)
    
    model_vgg7.save('Models/vgg7_dr35/e25/')
    model_vgg7.save_weights('Models/vgg7_dr35/e25/vgg7_model_weights_dr35.pb')

def model_2(ds_tr, ds_val, tr_length):
    drop_out = 0.25
    
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
            Dense(units=12, activation='relu', kernel_initializer='he_normal'),
            BatchNormalization(),
            Dropout(drop_out),
            Dense(units=4, activation='softmax', kernel_initializer='he_normal'),
    ])
    
    params = {'optimizer':['Adam'],
              'epochs':[25],
              'batch_size':[16],
              'learning_rate':[0.001]}
    params = ParameterGrid(params)
    
    history = History()
    
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
    plt.title("Vgg7 Loss (drop out=0.25)")
    plt.legend()
    plt.show()
    
    plt.plot(history.history['f1_score'], label='train')
    plt.plot(history.history['val_f1_score'], label='validation')
    plt.title("Vgg7 F1 Score (drop out=0.25)")
    plt.legend()
    plt.show()
    
    model_vgg7.evaluate(ds_tr)
    model_vgg7.evaluate(ds_val)
    
    model_vgg7.save('Models/vgg7_dr25/e25')
    model_vgg7.save_weights('Models/vgg7_dr25/e25/vgg7_model_weights_dr25.pb')
    
def model_3(ds_tr, ds_val, tr_length):
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
            Dense(units=12, activation='relu', kernel_initializer='he_normal'),
            BatchNormalization(),
            Dropout(drop_out),
            Dense(units=4, activation='softmax', kernel_initializer='he_normal'),
    ])
    
    history = History()
    
    params = {'optimizer':['Adam'],
              'epochs':[50],
              'batch_size':[16],
              'learning_rate':[0.001]}
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
    plt.title("Vgg7 Loss (drop out=0.35)")
    plt.legend()
    plt.show()
    
    plt.plot(history.history['f1_score'], label='train')
    plt.plot(history.history['val_f1_score'], label='validation')
    plt.title("Vgg7 F1 Score (drop out=0.35)")
    plt.legend()
    plt.show()
    
    model_vgg7.evaluate(ds_tr)
    model_vgg7.evaluate(ds_val)
    
    model_vgg7.save('Model/vgg7_dr35/e50')
    model_vgg7.save_weights('Models/vgg7_dr35/e50/vgg7_model_weights_dr35_e50.pb')
    
def model_4(ds_tr, ds_val, tr_length):
    drop_out = 0.25
    
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
            Dense(units=12, activation='relu', kernel_initializer='he_normal'),
            BatchNormalization(),
            Dropout(drop_out),
            Dense(units=4, activation='softmax', kernel_initializer='he_normal'),
    ])
    
    params = {'optimizer':['Adam'],
              'epochs':[50],
              'batch_size':[16],
              'learning_rate':[0.001]}
    params = ParameterGrid(params)
    
    history = History()
    
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
    plt.title("Vgg7 Loss (drop out=0.25)")
    plt.legend()
    plt.show()
    
    plt.plot(history.history['f1_score'], label='train')
    plt.plot(history.history['val_f1_score'], label='validation')
    plt.title("Vgg7 F1 Score (drop out=0.25)")
    plt.legend()
    plt.show()
    
    model_vgg7.evaluate(ds_tr)
    model_vgg7.evaluate(ds_val)
    
    model_vgg7.save('Models/vgg7_dr25/e50')
    model_vgg7.save_weights('Models/vgg7_dr25/e50/vgg7_model_weights_dr25_e50.pb')
    
def end_eval_model_4(ds_test, shuffled=False):
    '''As the model saves were added too late in development to be useful
    we redefine the model and just load the weights onto it. Shuffled (True/False) is
    whether the data is coming from a shuffle_on_iteration DataGen such as
    preprocessing.image_dataset_from_directory with shuffle=True
    '''
    drop_out = 0.25
    
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
            Dense(units=12, activation='relu', kernel_initializer='he_normal'),
            BatchNormalization(),
            Dropout(drop_out),
            Dense(units=4, activation='softmax', kernel_initializer='he_normal'),
    ])
    
    batch_size = 16
    optimizer = Adam(learning_rate=0.001)
    
    ds_test = ds_test.unbatch()
    ds_test = ds_test.batch(batch_size)
    
    model_vgg7 = keras.models.clone_model(model_vgg7)
    model_vgg7.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=[F1Score(num_classes=4, average='macro'),
                           CategoricalAccuracy()]
                  )
    
    path = 'Models/vgg7_dr25/e50/vgg7_model_weights_dr25_e50.pb'
    model_vgg7.load_weights(path)

    model_vgg7.evaluate(ds_test)
    
    if shuffled:
        #We have to treat this data differently as it is reshuffled every time
        #we iterate the ds_test object. If we aren't careful the images/preds
        #and labels become seperated as they are shuffled seperately.
        images = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        labels = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        
        for x, y in ds_test.unbatch():
          images = images.write(images.size(), x)
          labels = labels.write(labels.size(), y)
        
        images = tf.stack(images.stack(), axis=0)
        labels = tf.stack(labels.stack(), axis=0)
        
        
        with tf.device("/cpu:0"):
            #Cannot fit full tensor into gpu memory, which is why we normally
            #use a generator, so we run this pred on cpu.
            #Could have made custom generator function but as we only run this
            #once to get CM on validation data there's no point
            preds = model_vgg7.predict(images)
            
    else:
        labels = tf.convert_to_tensor(list(ds_test.unbatch().map(lambda x, y: y)))
        preds = model_vgg7.predict(ds_test)
    
    #Convert to argmax predictions for confusion matrix
    preds_argm = tf.math.argmax(preds, axis=1).numpy()
    labels_argm = tf.math.argmax(labels, axis=1).numpy()
    
    cm = confusion_matrix(labels_argm, preds_argm, labels=[0,1,2,3])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=['Benign','Early','Pre','Pro'])
    disp.plot()
    plt.show()

def load_and_preprocess():
    physical_devices = tf.config.list_physical_devices('GPU')
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except Exception:
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
            shuffle=False
            )
        
        class_names = ds_tr.class_names   
        print(f"\n Class names are {class_names}")
        
        tr_length = len(ds_tr)
        
        #Augmentation
        Flip = keras.layers.RandomFlip()
        Rotate = keras.layers.RandomRotation(0.25)
        Cont = keras.layers.RandomContrast(factor=0.1)

        
        def augment(image, label):
            image = Flip(image)
            image = Rotate(image)
            image = Cont(image)
            
            return image, label
        
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

        #Crop, then augment, then normalize    

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
        
        return ds_tr, ds_val, ds_test, tr_length

ds_tr, ds_val, ds_test, tr_length = load_and_preprocess()
model_1(ds_tr, ds_val, tr_length)
model_2(ds_tr, ds_val, tr_length)
model_3(ds_tr, ds_val, tr_length)
model_4(ds_tr, ds_val, tr_length)
end_eval_model_4(ds_val, shuffled=True)
end_eval_model_4(ds_test)