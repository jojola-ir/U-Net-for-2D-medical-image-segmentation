'''custom model building'''

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


def unet(n_levels, initial_features=32, n_blocks=2, kernel_size=3, pooling_size=2, in_channels=1, out_channels=1):
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
    IMAGE_HEIGHT = 320
    IMAGE_WIDTH = 320

    inputs = keras.layers.Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, in_channels))
    x = inputs

    convpars = dict(kernel_size=kernel_size, activation='relu', padding='same')

    # downstream
    skips = {}
    for level in range(n_levels):
        for _ in range(n_blocks):
            x = keras.layers.Conv2D(initial_features * 2 ** level, **convpars)(x)
        if level < n_levels - 1:
            skips[level] = x
            x = keras.layers.MaxPool2D(pooling_size)(x)

    # upstream
    for level in reversed(range(n_levels - 1)):
        x = keras.layers.Conv2DTranspose(initial_features * 2 ** level, strides=pooling_size, **convpars)(x)
        x = keras.layers.Concatenate()([x, skips[level]])
        for _ in range(n_blocks):
            x = keras.layers.Conv2D(initial_features * 2 ** level, **convpars)(x)

    # output
    activation = 'sigmoid' if out_channels == 1 else 'softmax'
    x = keras.layers.Conv2D(out_channels, kernel_size=1, activation=activation, padding='same')(x)

    return keras.Model(inputs=[inputs], outputs=[x], name=f'UNET-L{n_levels}-F{initial_features}')
