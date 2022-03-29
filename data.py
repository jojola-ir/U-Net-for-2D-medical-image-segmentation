'''Dataloader'''
import argparse
import glob
import os

import nibabel as nib
import numpy as np
import tensorflow as tf
from skimage import io
from tensorflow.keras.layers.experimental.preprocessing import (RandomFlip,
                                                                RandomHeight,
                                                                RandomRotation,
                                                                RandomWidth,
                                                                RandomZoom)
from tensorflow.keras.preprocessing.image import ImageDataGenerator

SLICE_X = True
SLICE_Y = True
SLICE_Z = False

SLICE_DECIMATE_IDENTIFIER = 3


def configure_for_performance(ds):
    """Adjusts training performance by putting datas in cache and by prefetch.

    Parameters
    ----------
    ds : tf.train.datasets
        Path to the dataset directory.

    Returns
    -------
    ds
        tf.train.datasets
    """

    ds = ds.cache()
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    return ds


def saveSlice(img, fname, path):
    # img = np.uint8(img * 255)
    fout = os.path.join(path, f'{fname}.png')
    io.imsave(fout, img)
    print(f'[+] Slice saved: {fout}', end='\r')


def sliceAndSaveVolumeImage(vol, fname, path):
    (dimx, dimy, dimz) = vol.shape
    print(dimx, dimy, dimz)
    cnt = 0
    if SLICE_X:
        cnt += dimx
        print('Slicing X: ')
        for i in range(dimx):
            saveSlice(vol[i, :, :], fname + f'-slice{str(i).zfill(SLICE_DECIMATE_IDENTIFIER)}_x', path)

    if SLICE_Y:
        cnt += dimy
        print('Slicing Y: ')
        for i in range(dimy):
            saveSlice(vol[:, i, :], fname + f'-slice{str(i).zfill(SLICE_DECIMATE_IDENTIFIER)}_y', path)

    if SLICE_Z:
        cnt += dimz
        print('Slicing Z: ')
        for i in range(dimz):
            saveSlice(vol[:, :, i], fname + f'-slice{str(i).zfill(SLICE_DECIMATE_IDENTIFIER)}_z', path)
    return cnt


def generate_dataset(path_to_images, path_to_output, organ="heart"):
    """Generates .png images from .nii images.

    Parameters
    ----------
    path_to_images : str
        Path to the .nii images.

    path_to_output : str
        Output path for .png images.

    normalize : boolean
        Enables or disables image normalisation.

    Returns
    -------
    None
    """

    imagePathInput = os.path.join(path_to_images, 'imagesTr/')
    maskPathInput = os.path.join(path_to_images, 'labelsTr/')

    imageSliceOutput = os.path.join(path_to_output, 'images/')
    maskSliceOutput = os.path.join(path_to_output, 'mask/')

    if os.path.exists(imageSliceOutput) is False:
        os.mkdir(imageSliceOutput)

    if os.path.exists(maskSliceOutput) is False:
        os.mkdir(maskSliceOutput)

    for index, filename in enumerate(sorted(glob.iglob(imagePathInput + '*.nii.gz'))):
        img = nib.load(filename).get_fdata()
        print(filename, img.shape, np.sum(img.shape), np.min(img), np.max(img))
        numOfSlices = sliceAndSaveVolumeImage(img, organ + str(index), imageSliceOutput)
        print(f'\n{filename}, {numOfSlices} slices created \n')

    for index, filename in enumerate(sorted(glob.iglob(maskPathInput + '*.nii.gz'))):
        img = nib.load(filename).get_fdata()
        print(filename, img.shape, np.sum(img.shape), np.min(img), np.max(img))
        numOfSlices = sliceAndSaveVolumeImage(img, organ + str(index), maskSliceOutput)
        print(f'\n{filename}, {numOfSlices} slices created \n')


def create_pipeline(path, performance=False, bs=256):
    """Creates datasets from a directory given as parameter.

    The set given as input must include training and validation directory.

    Parameters
    ----------
    path_to_dataset : str
        Path to the dataset directory.

    performance : boolean (False by default)
        Enables or disables performance configuration.

    bs : int
        Batch size

    Returns
    -------
    train_dataset, val_dataset, test_dataset
        datasets for training, validating and testing
    """

    data_gen_args = dict(rescale=1. / 255,
                         # featurewise_center=True,
                         # featurewise_std_normalization=True,
                         # rotation_range=90,
                         # width_shift_range=0.2,
                         # height_shift_range=0.2,
                         # zoom_range=0.3
                         )

    img_height = 320
    img_width = 320

    datagen = ImageDataGenerator(**data_gen_args, validation_split=0.2)

    print("Creating the full train dataset")
    train_image_generator = datagen.flow_from_directory(
        os.path.join(path, "images/"),
        target_size=(img_height, img_width),
        class_mode=None,
        color_mode='grayscale',
        batch_size=bs,
        seed=0,
        subset="training")

    train_mask_generator = datagen.flow_from_directory(
        os.path.join(path, "masks/"),
        target_size=(img_height, img_width),
        class_mode=None,
        color_mode='grayscale',
        batch_size=bs,
        seed=0,
        subset="training")

    test_image_generator = datagen.flow_from_directory(
        os.path.join(path, "images/"),
        target_size=(img_height, img_width),
        batch_size=bs,
        subset="validation")

    test_mask_generator = datagen.flow_from_directory(
        os.path.join(path, "masks/"),
        target_size=(img_height, img_width),
        batch_size=bs,
        subset="validation")

    train_pipeline = zip(train_image_generator, train_mask_generator)
    test_pipeline = zip(test_image_generator, test_mask_generator)

    # train_dataset = full_pipeline.skip(10)
    # val_dataset = full_pipeline.take(10)

    if performance:
        train_pipeline = configure_for_performance(train_pipeline)
        test_pipeline = configure_for_performance(test_pipeline)

    # return train_dataset, val_dataset
    return train_pipeline, test_pipeline


def preprocess(ds):
    return ds


def augmentation():
    """Adds data augmentation preprocessing."""

    data_augmentation = tf.keras.Sequential()
    data_augmentation.add(RandomFlip("horizontal_and_vertical"))
    data_augmentation.add(RandomRotation(0.2))
    data_augmentation.add(RandomZoom(
        height_factor=(-0.3, 0.3), width_factor=(-0.3, 0.3)))
    data_augmentation.add(RandomHeight(
        factor=(0.2, 0.3), interpolation="bicubic"))
    data_augmentation.add(RandomWidth(
        factor=(0.2, 0.3), interpolation="bicubic"))

    return data_augmentation


def main():
    """This is the main function"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--generate", default=False, help="generate ")
    parser.add_argument("--batch", "-b", type=int,
                        help="batch size")
    parser.add_argument("--datapath", help="path to the dataset")
    parser.add_argument("--output", help="path to the output")

    args = parser.parse_args()

    datapath = args.datapath

    data_path = os.path.join(datapath)
    print(os.path.isdir(data_path))

    if args.generate:
        output = args.output
        generate_dataset(datapath, output)

    else:
        if args.batch:
            bs = args.batch
        else:
            bs = 32

        # some manipulation on the data
        train_set, test_set = create_pipeline(datapath, bs=bs)

        for img, mask in train_set.take(1):
            print("One batch of data")
            print(img.shape, mask.shape)


if __name__ == "__main__":
    main()
