from keras import Input, Model
from keras.layers import Conv2D
from keras.src.layers import MaxPooling2D, Conv2DTranspose
from numpy import concatenate
import numpy as np
import random as rn
import cv2 as cv

from Evaluate_Error import evaluat_error


def unet_model(input_shape=(256, 256, 3)):
    inputs = Input(shape=input_shape)

    # Encoder
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, 3, activation='relu', padding='same')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same')(conv5)

    # Decoder
    up6 = Conv2DTranspose(512, 2, strides=(2, 2), padding='same')(conv5)
    up6 = concatenate([up6, conv4], axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same')(up6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same')(conv6)

    up7 = Conv2DTranspose(256, 2, strides=(2, 2), padding='same')(conv6)
    up7 = concatenate([up7, conv3], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same')(up7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same')(conv7)

    up8 = Conv2DTranspose(128, 2, strides=(2, 2), padding='same')(conv7)
    up8 = concatenate([up8, conv2], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same')(up8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same')(conv8)

    up9 = Conv2DTranspose(64, 2, strides=(2, 2), padding='same')(conv8)
    up9 = concatenate([up9, conv1], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same')(up9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same')(conv9)

    outputs = Conv2D(1, 1, activation='sigmoid')(conv9)

    model = Model(inputs=inputs, outputs=outputs)

    return model

def Mode_Unet(train_data, train_target, test_data,test_target):
    # Set some parameters
    IMG_SIZE = 256
    seed = 42
    rn.seed = seed
    np.random.seed = seed

    X_train = np.zeros((train_data.shape[0], IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    Y_train = np.zeros((train_target.shape[0], IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    X_test = np.zeros((test_data.shape[0], IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)

    for i in range(train_data.shape[0]):
        Temp = cv.resize(train_data[i, :], (IMG_SIZE, IMG_SIZE))
        X_train[i, :, :, :] = Temp.reshape((IMG_SIZE, IMG_SIZE, 3))

    for i in range(train_target.shape[0]):
        Temp = cv.resize(train_target[i, :], (IMG_SIZE, IMG_SIZE))
        Temp = Temp.reshape((IMG_SIZE, IMG_SIZE, 3))
        for j in range(Temp.shape[0]):
            for k in range(Temp.shape[1]):
                if Temp[j, k] < 0.5:
                    Temp[j, k] = 0
                else:
                    Temp[j, k] = 1
        Y_train[i, :, :, :] = Temp

    for i in range(test_data.shape[0]):
        Temp = cv.resize(test_data[i, :], (IMG_SIZE, IMG_SIZE))
        X_test[i, :, :, :] = Temp.reshape((IMG_SIZE, IMG_SIZE, 3))
    model = unet_model()
    model.summary()
    model.fit(X_train, Y_train)
    pred_img = model.predict(X_test)
    ret_img = pred_img[:, :, :, 0]
    Eval = evaluat_error(ret_img,test_target)
    return Eval

