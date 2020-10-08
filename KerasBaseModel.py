# -*- coding:utf-8 -*-
from tensorflow.python.keras.models import Model, load_model
import numpy as np
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.layers import Convolution2D
from tensorflow.python.keras.layers import Dropout, Flatten, Dense
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.python.keras.applications.mobilenet import MobileNet
import datetime

#驾驶模型调用
class DriveModel():
    def __init__(self, model_path=None):
    	self.model_path = model_path

    	if model_path != None:
            self.model = None
            self.load()
    	else:
    		self.model = default_model()

    def load(self):
    	self.model = load_model(self.model_path)
    	print("----Load the model["+ self.model_path +"] successfully----")

    def predict_image(self, frame):
    	img_arr = frame.reshape((1,) + frame.shape)
    	angle_binned, throttle = self.model.predict(img_arr)
    	angle_unbinned = self.linear_unbin(angle_binned[0])

    	return angle_unbinned, throttle[0][0]

    def linear_unbin(self, arr):
        # Convert a categorical array to value.
        if not len(arr) == 15:
            raise ValueError('Illegal array length, must be 15')
        b = np.argmax(arr)
        a = b * (2 / 14) - 1
        return a

    def train(self, train_gen, val_gen,
              saved_model_path, epochs=100, steps=100, train_split=0.8,
              verbose=1, min_delta=.0005, patience=5, use_early_stop=True):
        """
        train_gen: generator that yields an array of images an array of

        """

        # checkpoint to save model after each epoch
        save_best = ModelCheckpoint(saved_model_path,
                                    monitor='val_loss',
                                    verbose=verbose,
                                    save_best_only=True,
                                    mode='min')

        # stop training if the validation error stops improving.
        early_stop = EarlyStopping(monitor='val_loss',
                                   min_delta=min_delta,
                                   patience=patience,
                                   verbose=verbose,
                                   mode='auto')
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=2, min_lr=0.001)

        callbacks_list = [save_best]
        callbacks_list.append(reduce_lr)

        if use_early_stop:
            callbacks_list.append(early_stop)
        start = datetime.datetime.now()   
        hist = self.model.fit_generator(
            train_gen,
            steps_per_epoch=steps,
            epochs=epochs,
            verbose=1,
            validation_data=val_gen,
            callbacks=callbacks_list,
            validation_steps=int(steps * (1.0 - train_split) / train_split))
        end = datetime.datetime.now()
        print('TRAIN TIME:',end-start)
        return hist
     
    

def default_model():
    img_in = Input(shape=(224, 224, 3),name='img_in')  
    '''
    x = img_in
    x = Convolution2D(24, (5, 5), strides=(2, 2),activation='relu')(x)
    
    x = Convolution2D(32, (3, 3), strides=(2, 2),padding='same',activation='relu')(x)
    
    x = Convolution2D(64, (3, 3), strides=(2, 2),padding='same',activation='relu')(x)
    x = Convolution2D(64, (3, 3), strides=(1, 1),padding='same',activation='relu')(x)
    
    x = Convolution2D(64, (3, 3), strides=(2, 2),padding='same',activation='relu')(x)
    x = Convolution2D(64, (3, 3), strides=(1, 1),padding='same',activation='relu')(x)
    
    x = Convolution2D(128, (3, 3), strides=(2, 2), padding='same',activation='relu')(x)
    x = Convolution2D(128, (3, 3), strides=(1, 1), padding='same',activation='relu')(x)
    
    x = Flatten(name='flattened')(x)  
    x = Dense(100, activation='relu')(x)  
    x = Dropout(.1)(x) 
    x = Dense(50, activation='relu')(x)  
    x = Dropout(.1)(x) 
    '''
    base_model = MobileNet(input_shape=None, alpha=1.0, depth_multiplier=1, dropout=1e-3, 
                           include_top=True, weights='imagenet', input_tensor=None, pooling='max', classes=1000)
    
    x = base_model.output
    
    angle_out = Dense(15, activation='softmax', name='angle_out')(x) 

    throttle_out = Dense(1, activation='relu', name='throttle_out')(x)  

    model = Model(inputs=[img_in], outputs=[angle_out, throttle_out])
    model.compile(optimizer='adam',
                  loss={'angle_out': 'categorical_crossentropy',
                        'throttle_out': 'mean_absolute_error'},
                  loss_weights={'angle_out': 0.9, 'throttle_out': 0.1},
                  metrics=["accuracy"]                 
                  )
    
    return model

