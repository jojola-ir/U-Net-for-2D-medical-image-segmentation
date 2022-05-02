'''custom model building'''

import argparse
import os
from datetime import datetime

from tensorflow import keras
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers.experimental.preprocessing import Rescaling

from data import augmentation


def custom_model(clNbr, pw, da=False):
    """Build a convolutional neural network composed of resnet50 pretrained
    model and a fully connected layer.

    The number of output neurons is automatically set to the number of classes
    that make up the dataset.

    Parameters
    ----------
    clNbr : int
        Number of classes that defines output

    da: boolean
        Enables or disable data augmentation

    Returns
    -------
    model
        builded model
    """

    model = keras.models.Sequential()
    model.add(Rescaling(scale=1. / 255))
    if da:
        aug = augmentation()
        model.add(aug)
    model.add(ResNet50(include_top=True, weights=pw))
    model.add(keras.layers.Dense(clNbr, activation="softmax"))

    return model


def unet(n_levels, initial_features=64, n_blocks=2, kernel_size=3, pooling_size=2, in_channels=1, out_channels=1):
    """Build a neural network composed of UNET architecture.

    Parameters
    ----------
    clNbr : int
        Number of classes that defines output

    da: boolean
        Enables or disable data augmentation

    Returns
    -------
    model
        builded model
    """
    IMAGE_HEIGHT = 160
    IMAGE_WIDTH = 160

    inputs = keras.layers.Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, in_channels))
    x = inputs

    convpars = dict(kernel_size=kernel_size, activation='relu', padding='same')

    # downstream
    skips = {}
    for level in range(n_levels):
        for block in range(n_blocks):
            x = keras.layers.SeparableConv2D(initial_features * 2 ** level, **convpars)(x)
            if level <= n_levels // 2 and block == 0:
                x = keras.layers.Dropout(0.1)(x)
            elif level > n_levels // 2 and level < n_levels - 1 and block == 0:
                x = keras.layers.Dropout(0.2)(x)
        if level < n_levels - 1:
            skips[level] = x
            x = keras.layers.MaxPool2D(pooling_size)(x)

    # upstream
    for level in reversed(range(n_levels - 1)):
        x = keras.layers.Conv2DTranspose(initial_features * 2 ** level, strides=pooling_size, **convpars)(x)
        x = keras.layers.Concatenate()([x, skips[level]])
        for block in range(n_blocks):
            x = keras.layers.SeparableConv2D(initial_features * 2 ** level, **convpars)(x)
            if level > n_levels // 2 and block == 0:
                x = keras.layers.Dropout(0.1)(x)
            elif level <= n_levels // 2 and block == 0:
                x = keras.layers.Dropout(0.2)(x)

    # output
    activation = 'sigmoid' if out_channels == 1 else 'softmax'
    x = keras.layers.SeparableConv2D(out_channels, kernel_size=1, activation=activation, padding='same')(x)

    return keras.Model(inputs=[inputs], outputs=[x], name=f'UNET-L{n_levels}-F{initial_features}')


def multi_task_unet(n_levels, initial_features=32, n_blocks=2,kernel_size=3,
                    pooling_size=2, in_channels=1, out_channels=1, reconstruction=False,
                    segmentation=False, custom_weights=None, merge=False):
    """Build a neural network composed of UNET architecture.

    Parameters
    ----------
    clNbr : int
        Number of classes that defines output

    da: boolean
        Enables or disable data augmentation

    Returns
    -------
    model
        builded model
    """
    IMAGE_HEIGHT = 160
    IMAGE_WIDTH = 160

    inputs = keras.layers.Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, in_channels))
    x = inputs

    encoder_layers = 1

    convpars = dict(kernel_size=kernel_size, activation="relu", padding="same")

    # downstream
    skips = {}
    for level in range(n_levels):
        for block in range(n_blocks):
            x = keras.layers.SeparableConv2D(initial_features * 2 ** level, **convpars)(x)
            encoder_layers += 1
            if level <= n_levels // 2 and block == 0:
                x = keras.layers.BatchNormalization()(x)
                x = keras.layers.Dropout(0.3)(x)
                encoder_layers += 1
            elif level > n_levels // 2 and level < n_levels - 1 and block == 0:
                x = keras.layers.BatchNormalization()(x)
                x = keras.layers.Dropout(0.5)(x)
                encoder_layers += 1
        if level < n_levels - 1:
            skips[level] = x
            x = keras.layers.MaxPool2D(pooling_size, padding="same")(x)
            encoder_layers += 1

    if reconstruction:
        # upstream : reconstruction
        for level in reversed(range(n_levels - 1)):
            if level == n_levels - 2:
                x_r = keras.layers.Conv2DTranspose(initial_features * 2 ** level, strides=pooling_size, **convpars)(x)
            else:
                x_r = keras.layers.Conv2DTranspose(initial_features * 2 ** level, strides=pooling_size, **convpars)(x_r)
            x_r = keras.layers.Concatenate()([x_r, skips[level]])
            for block in range(n_blocks):
                x_r = keras.layers.SeparableConv2D(initial_features * 2 ** level, **convpars)(x_r)
                if level < n_levels // 2 and block == 0:
                    x_r = keras.layers.BatchNormalization()(x_r)
                    #x_r = keras.layers.Dropout(0.1)(x_r)

    if segmentation:
        # upstream : segmentation
        for level in reversed(range(n_levels - 1)):
            x = keras.layers.Conv2DTranspose(initial_features * 2 ** level, strides=pooling_size, **convpars)(x)
            x = keras.layers.Concatenate()([x, skips[level]])
            for block in range(n_blocks):
                x = keras.layers.SeparableConv2D(initial_features * 2 ** level, **convpars)(x)
                if level > n_levels // 2 and block == 0:
                    x = keras.layers.BatchNormalization()(x)
                    x = keras.layers.Dropout(0.3)(x)
                elif level <= n_levels // 2 and block == 0:
                    x = keras.layers.BatchNormalization()(x)
                    x = keras.layers.Dropout(0.5)(x)

    # outputs
    activation = "sigmoid" if out_channels == 1 else "softmax"
    outputs = []
    if reconstruction:
        x_r = keras.layers.SeparableConv2D(out_channels, kernel_size=1, activation=activation, padding="same")(x_r)
        outputs.append(x_r)

    if segmentation:
        x = keras.layers.SeparableConv2D(out_channels, kernel_size=1, activation=activation, padding="same")(x)
        outputs.append(x)

    model = keras.Model(inputs=[inputs], outputs=outputs, name=f"UNET-L{n_levels}-F{initial_features}")

    if custom_weights != None:
        print(f"Number of encoder layers : {encoder_layers}")
        for layers, pretrained_layers in zip(model.layers[:encoder_layers], custom_weights.layers[:encoder_layers]):
            layers.set_weights(pretrained_layers.get_weights())

    if merge:
        print("Merging reconstruction and segmentation models")

    return model


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--reconstruction", "-r", default=False,
                        help="train model to reconstruct input image",
                        action="store_true")
    parser.add_argument("--segmentation", "-s", default=False,
                        help="train model to segment input image",
                        action="store_true")
    parser.add_argument("--merge", "-m", default=False,
                        help="merge reconstruction and segmentation pre-trained model",
                        action="store_true")
    parser.add_argument("--load", default=False,
                        help="load previous model",
                        action="store_true")
    parser.add_argument("--modelpath", default="/models/run1.h5",
                        help="path to .h5 file for transfert learning")

    args = parser.parse_args()
    reconstruction = args.reconstruction
    segmentation = args.segmentation
    merge = args.merge
    load_model = args.load
    if load_model:
        model_name = "custom"
        model_path = args.modelpath
        custom_model = keras.models.load_model(model_path, compile=False)
        print(f"Transfert learning from {model_path}")
        model = multi_task_unet(5, reconstruction=reconstruction, segmentation=segmentation, custom_model=custom_model)
    else:
        model = multi_task_unet(5, initial_features=32, reconstruction=reconstruction, segmentation=segmentation)

    model.summary()

    model_architecture_path = "architecture/test/"
    if os.path.exists(model_architecture_path) is False:
        os.makedirs(model_architecture_path)

    now = datetime.now()
    keras.utils.plot_model(model,
                           to_file=os.path.join(model_architecture_path,
                                                f"model_unet_{now.strftime('%m_%d_%H_%M')}.png"),
                           show_shapes=True)