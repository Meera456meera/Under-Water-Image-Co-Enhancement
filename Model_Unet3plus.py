from keras import Input, Model
from keras.src.layers import Conv2D, BatchNormalization, Activation, UpSampling2D, Concatenate, MaxPooling2D
import numpy as np
import random as rn
import cv2 as cv
from Evaluate_Error import evaluat_error
def conv_block(inputs, filters, kernel_size=(3, 3), padding='same', activation='relu'):
    conv = Conv2D(filters, kernel_size, padding=padding)(inputs)
    conv = BatchNormalization()(conv)
    conv = Activation(activation)(conv)
    return conv

def upsample_concat_block(inputs, skip_inputs, filters, kernel_size=(3, 3), padding='same', activation='relu'):
    upsampled = UpSampling2D(size=(2, 2))(inputs)
    concat = Concatenate()([upsampled, skip_inputs])
    conv = conv_block(concat, filters, kernel_size, padding, activation)
    return conv

def UNet3Plus_model(input_shape=(256, 256, 3), num_classes=1):
    inputs = Input(shape=input_shape)

    # Encoder (downsampling path)
    conv1 = conv_block(inputs, 64)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = conv_block(pool1, 128)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = conv_block(pool2, 256)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    # Bridge
    bridge = conv_block(pool3, 512)

    # Decoder (upsampling path)
    up3 = upsample_concat_block(bridge, conv3, 256)
    up2 = upsample_concat_block(up3, conv2, 128)
    up1 = upsample_concat_block(up2, conv1, 64)

    # Output
    outputs = Conv2D(num_classes, (1, 1), activation='sigmoid')(up1)

    model = Model(inputs=[inputs], outputs=[outputs])

    return model

def Model_Unet3plus(train_data, train_target, test_data,test_target):
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
    model = UNet3Plus_model()
    model.summary()
    model.fit(X_train, Y_train)
    pred_img = model.predict(X_test)
    ret_img = pred_img[:, :, :, 0]
    Eval = evaluat_error(ret_img, test_target)
    return Eval
