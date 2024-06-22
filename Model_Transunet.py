import numpy as np
import random as rn
import cv2 as cv
import torch.nn.functional as F
import torch.nn as nn
from keras.src.losses import BinaryCrossentropy
from keras.src.optimizers import Adam

from Evaluate_Error import evaluat_error


class TransUNet(nn.Module):
    def __init__(self,input_shape, num_classes, num_layers=12, num_heads=12, patch_size=16):
        super(TransUNet, self).__init__()

        # Transformer Encoder
        self.encoder = TransformerEncoder(num_layers, num_heads)

        # U-Net Decoder
        self.decoder = UNetDecoder(num_classes, patch_size)

    def forward(self, x):
        # Encoder
        encoded = self.encoder(x)

        # Decoder
        decoded = self.decoder(encoded)

        return decoded


class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, num_heads):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=..., nhead=num_heads)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class UNetDecoder(nn.Module):
    def __init__(self, num_classes, patch_size):
        super(UNetDecoder, self).__init__()

        # Define your U-Net decoder layers here
        # Example:
        self.conv1 = nn.Conv2d(in_channels=..., out_channels=..., kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=..., out_channels=..., kernel_size=3, padding=1)
        ...

    def forward(self, x):
        # Define the forward pass of your U-Net decoder
        # Example:
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        ...

        return x


def transunet_segmentation(sol,input_shape=(256, 256, 3), num_classes=1):
    # Create TransUNet model
    model = TransUNet( input_shape,num_classes, num_layers=12, num_heads=sol[0], patch_size=16)
    model.compile(optimizer=Adam(lr=1e-4),loss=BinaryCrossentropy(),metrics=['accuracy'] )
    return model

def Model_TransUnet(train_data, train_target, test_data,test_target,sol=None):
    if sol is None:
        sol = [5,5,300]
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
    model = transunet_segmentation(sol)
    model.summary()
    model.fit(X_train, Y_train,epoch = sol[1],Steps_per_epoch = sol[2])
    pred_img = model.predict(X_test)
    ret_img = pred_img[:, :, :, 0]
    Eval = evaluat_error(ret_img, test_target)
    return Eval, ret_img
