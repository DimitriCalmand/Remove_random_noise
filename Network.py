import os
from PIL import Image as img
import matplotlib.pyplot as plt
import numpy as np
import cv2
import tensorflow as tf
from keras import layers
from keras.models import Model,Sequential

def downSample (num_filer,stride,dropout = False):
    initializer = tf.random_normal_initializer(0., 0.02)
    model = Sequential()
    model.add(layers.Conv2D(num_filer,(4,4),stride,padding="same",use_bias=False,kernel_initializer=initializer))  
    model.add(layers.BatchNormalization())
    if dropout :
        model.add(layers.Dropout(0.2))
    model.add(layers.LeakyReLU(0.3))   
    return model
def upSample(num_filter,stride,dropout = False):
    initializer = tf.random_normal_initializer(0., 0.02)
    model = Sequential()
    model.add(layers.Conv2DTranspose(num_filter,(4,4),stride,padding="same",use_bias=False,kernel_initializer=initializer))
    model.add(layers.BatchNormalization())
    if dropout :
        model.add(layers.Dropout(0.5))
    model.add(layers.Activation(layers.ReLU()))
    return model

def maxPool(num_filter):
    return downSample(num_filter,2)
def generator ():
    input = layers.Input(shape = (256,256,3))

    first_conv1 = downSample(64,1)(input)
    first_conv2 = downSample(64,1)(first_conv1)
    first_conv3 = maxPool(64)(first_conv2)

    # Model size = (128,128,64)
    
    second_conv1 = downSample(128,1)(first_conv3)
    second_conv2 = downSample(128,1)(second_conv1)
    second_conv3 = maxPool(128)(second_conv2)

    # Model size = (64,64,128)
    
    third_conv1 = downSample(256,1)(second_conv3)
    third_conv2 = downSample(256,1)(third_conv1)
    third_conv3 = maxPool(256)(third_conv2)

    # Model size = (32,32,256)

    fourth_conv1 = downSample(512,1)(third_conv3)
    fourth_conv2 = downSample(512,1)(fourth_conv1)
    fourth_conv3 = maxPool(512)(fourth_conv2)

    fiveth_conv1 = downSample(1024,1)(fourth_conv3)
    fiveth_conv2 = downSample(1024,1)(fiveth_conv1)


    zero_convtranspose1 = layers.concatenate([fourth_conv2,upSample(512,2,dropout=True)(fiveth_conv2)])
    zero_convtranspose2 = upSample(256,1)(zero_convtranspose1)
    zero_convtranspose3 = upSample(256,1)(zero_convtranspose2)

    # Model size = (32,32,1024)
    first_convtranspose1 = layers.concatenate([third_conv2,upSample(256,2,dropout=True)(zero_convtranspose3)])
    first_convtranspose2 = upSample(256,1)(first_convtranspose1)
    first_convtranspose3 = upSample(256,1)(first_convtranspose2)

    second_convtranspose1 = layers.concatenate([second_conv2,upSample(128,2,dropout=True)(first_convtranspose3)])
    second_convtranspose2 = upSample(128,1)(second_convtranspose1)
    second_convtranspose3 = upSample(128,1)(second_convtranspose2)

    third_convtranspose1 = layers.concatenate([first_conv2,upSample(64,2,dropout=True)(second_convtranspose3)])
    third_convtranspose2 = upSample(64,1)(third_convtranspose1)
    third_convtranspose3 = upSample(64,1)(third_convtranspose2)

    last = layers.Conv2DTranspose(3,4,1,"same",activation="sigmoid")(third_convtranspose3)

    return Model(input,last)

def discrim():
    input = layers.Input((256,256,3))

    output = layers.Conv2D(64,(3,3),2,"same")(input)
    output = layers.Dropout(0.2)(output)
    output = layers.BatchNormalization()(output)
    output = layers.Activation(layers.LeakyReLU())(output)
    
    # Model Size = (193)
    output = layers.Conv2D(128,(3,3),2,"same")(output)

    output = layers.Dropout(0.2)(output)
    output = layers.BatchNormalization()(output)
    output = layers.Activation(layers.LeakyReLU())(output)
    

    output = layers.Conv2D(256,(3,3),2,"same")(output)
    output = layers.Dropout(0.2)(output)
    output = layers.BatchNormalization()(output)
    output = layers.Activation(layers.LeakyReLU())(output)
    

    output = layers.Conv2D(256,(3,3),2,"same")(output)

    output = layers.Dropout(0.2)(output)
    output = layers.BatchNormalization()(output)
    output = layers.Activation(layers.LeakyReLU())(output)
    

    output = layers.Conv2D(512,(3,3),2,"same")(output)
    output = layers.Dropout(0.2)(output)
    output = layers.BatchNormalization()(output)
    output = layers.Activation(layers.LeakyReLU())(output)  

    output = layers.Flatten()(output)
    output = layers.Dense(1024,activation=layers.LeakyReLU())(output)
    # output = layers.Dense(100,"sigmoid")(output)
    output = layers.Dense(1,"sigmoid")(output)

    return Model(input,output)
# a = generator()
# print(a.summary())