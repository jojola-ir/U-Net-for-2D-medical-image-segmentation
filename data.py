'''Dataloader'''
import argparse
import os
from glob import glob, iglob

import imageio.core.util
import nibabel as nib
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
SLICE_Z = True

SLICE_DECIMATE_IDENTIFIER = 3


def silence_imageio_warning(*args, **kwargs):
    pass


imageio.core.util._precision_warn = silence_imageio_warning


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
    io.imsave(fout, img, check_contrast=False)
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


def generate_dataset(path_to_images, path_to_output, organ="heart", test=False):
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

    if test:
        imagePathInput = os.path.join(path_to_images, 'imagesTs/')

        imageSliceOutput = os.path.join(path_to_output, 'images/')

        for root, _, files in os.walk(imagePathInput):
            for f in files:
                print(f)

        if os.path.exists(imageSliceOutput) is False:
            os.makedirs(imageSliceOutput)

        for index, filename in enumerate(sorted(iglob(imagePathInput + '*.nii.gz'))):
            img = nib.load(filename).get_fdata()
            print(filename, img.shape, np.sum(img.shape), np.min(img), np.max(img))
            numOfSlices = sliceAndSaveVolumeImage(img, organ + str(index), imageSliceOutput)
            print(f'\n{filename}, {numOfSlices} slices created \n')

    else:
        imagePathInput = os.path.join(path_to_images, 'imagesTr/')
        maskPathInput = os.path.join(path_to_images, 'labelsTr/')

        imageSliceOutput = os.path.join(path_to_output, 'images/')
        maskSliceOutput = os.path.join(path_to_output, 'mask/')

        if os.path.exists(imageSliceOutput) is False:
            os.makedirs(imageSliceOutput)

        if os.path.exists(maskSliceOutput) is False:
            os.makedirs(maskSliceOutput)

        for index, filename in enumerate(sorted(iglob(imagePathInput + '*.nii.gz'))):
            img = nib.load(filename).get_fdata()
            print(filename, img.shape, np.sum(img.shape), np.min(img), np.max(img))
            numOfSlices = sliceAndSaveVolumeImage(img, organ + str(index), imageSliceOutput)
            print(f'\n{filename}, {numOfSlices} slices created \n')

        for index, filename in enumerate(sorted(iglob(maskPathInput + '*.nii.gz'))):
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

    train_data_gen_args = dict(rescale=1. / 255,
                               fill_mode='reflect',
                               rotation_range=90,
                               width_shift_range=0.2,
                               height_shift_range=0.2,
                               zoom_range=0.3
                               )

    test_data_gen_args = dict(rescale=1. / 255,
                              rotation_range=90,
                              width_shift_range=0.2,
                              height_shift_range=0.2,
                              zoom_range=0.3,
                              fill_mode='reflect'
                              )

    img_height = 160
    img_width = 160

    train_datagen = ImageDataGenerator(**train_data_gen_args, validation_split=0.2)
    test_datagen = ImageDataGenerator(**test_data_gen_args, validation_split=0.2)

    print("Creating the full train dataset")
    train_image_generator = train_datagen.flow_from_directory(
        os.path.join(path, "train/images/"),
        target_size=(img_height, img_width),
        class_mode=None,
        color_mode='grayscale',
        batch_size=bs,
        seed=0,
        subset="training")

    train_mask_generator = train_datagen.flow_from_directory(
        os.path.join(path, "train/masks/"),
        target_size=(img_height, img_width),
        class_mode=None,
        color_mode='grayscale',
        batch_size=bs,
        seed=0,
        subset="training")

    val_image_generator = train_datagen.flow_from_directory(
        os.path.join(path, "train/images/"),
        target_size=(img_height, img_width),
        class_mode=None,
        color_mode='grayscale',
        batch_size=bs,
        seed=202,
        subset="validation")

    val_mask_generator = train_datagen.flow_from_directory(
        os.path.join(path, "train/masks/"),
        target_size=(img_height, img_width),
        class_mode=None,
        color_mode='grayscale',
        batch_size=bs,
        seed=202,
        subset="validation")

    test_image_generator = test_datagen.flow_from_directory(
        os.path.join(path, "test/images/"),
        target_size=(img_height, img_width),
        class_mode=None,
        color_mode='grayscale',
        seed=909,
        batch_size=bs)

    test_mask_generator = test_datagen.flow_from_directory(
        os.path.join(path, "test/masks/"),
        target_size=(img_height, img_width),
        class_mode=None,
        color_mode='grayscale',
        seed=909,
        batch_size=bs)

    train_pipeline = zip(train_image_generator, train_mask_generator)
    val_pipeline = zip(val_image_generator, val_mask_generator)
    test_pipeline = zip(test_image_generator, test_mask_generator)

    # train_dataset = full_pipeline.skip(10)
    # val_dataset = full_pipeline.take(10)

    if performance:
        train_pipeline = configure_for_performance(train_pipeline)
        val_pipeline = configure_for_performance(val_pipeline)
        test_pipeline = configure_for_performance(test_pipeline)

    # return train_dataset, val_dataset
    return train_pipeline, val_pipeline, test_pipeline


def parse_image(path):
    """Load an image and its annotation (mask) and returning
    a dictionary.

    Parameters
    ----------
    img_path : str
        Image (not the mask) location.

    Returns
    -------
    dict
        Dictionary mapping an image and its annotation.
    """

    mask_path = tf.strings.regex_replace(path, "images", "masks")

    image = tf.io.read_file(path)
    image = tf.image.decode_png(image, channels=1)
    image = tf.image.convert_image_dtype(image, tf.uint8)

    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)

    return {'image': image, 'mask': mask}


def parse_image_reconstruction(path):
    """Load an image and its annotation (mask) and returning
    a dictionary.

    Parameters
    ----------
    img_path : str
        Image (not the mask) location.

    Returns
    -------
    dict
        Dictionary mapping an image and its annotation.
    """

    image = tf.io.read_file(path)
    image = tf.image.decode_png(image, channels=1)
    image = tf.image.convert_image_dtype(image, tf.uint8)

    return {'image': image, 'mask': image}


@tf.function
def normalize(input_image, input_mask):
    """Rescale the pixel values of the images between 0.0 and 1.0
    compared to [0,255] originally.

    Parameters
    ----------
    input_image : tf.Tensor
        Tensorflow tensor containing an image of size [SIZE,SIZE,3].
    input_mask : tf.Tensor
        Tensorflow tensor containing an annotation of size [SIZE,SIZE,1].

    Returns
    -------
    tuple
        Normalized image and its annotation.
    """
    input_image = tf.cast(input_image, tf.float32) / 255.0
    input_mask = tf.cast(input_mask, tf.float32) / 255.0

    return input_image, input_mask


@tf.function
def load_image_train(datapoint):
    """Apply some transformations to an input dictionary
    containing a train image and its annotation.

    Notes
    -----
    An annotation is a regular  channel image.
    If a transformation such as rotation is applied to the image,
    the same transformation has to be applied on the annotation also.

    Parameters
    ----------
    datapoint : dict
        A dict containing an image and its annotation.

    Returns
    -------
    tuple
        A modified image and its annotation.
    """

    IMG_SIZE = 160

    input_image = tf.image.resize_with_pad(datapoint['image'], IMG_SIZE, IMG_SIZE)
    input_mask = tf.image.resize_with_pad(datapoint['mask'], IMG_SIZE, IMG_SIZE)

    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        input_mask = tf.image.flip_left_right(input_mask)

    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_up_down(input_image)
        input_mask = tf.image.flip_up_down(input_mask)

    if tf.random.uniform(()) > 0.5:
        rd = np.random.random()
        input_image = tf.image.central_crop(input_image, central_fraction=rd)
        input_mask = tf.image.central_crop(input_mask, central_fraction=rd)

        input_image = tf.image.resize_with_pad(input_image, IMG_SIZE, IMG_SIZE)
        input_mask = tf.image.resize_with_pad(input_mask, IMG_SIZE, IMG_SIZE)

    input_image = tf.image.random_brightness(input_image, 0.3)
    input_image = tf.image.random_contrast(input_image, 0.2, 0.5)

    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask


@tf.function
def load_image_test(datapoint):
    """Normalize and resize a test image and its annotation.

    Notes
    -----
    Since this is for the test set, we don't need to apply
    any data augmentation technique.

    Parameters
    ----------
    datapoint : dict
        A dict containing an image and its annotation.

    Returns
    -------
    tuple
        A modified image and its annotation.
    """

    IMG_SIZE = 160

    input_image = tf.image.resize_with_pad(datapoint['image'], (IMG_SIZE, IMG_SIZE))
    input_mask = tf.image.resize_with_pad(datapoint['mask'], (IMG_SIZE, IMG_SIZE))

    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask


def create_pipeline_performance(path, bs=256, reconstruction=False, test=False):
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

    TRAIN_SEED = 202
    VAL_SEED = 505
    TEST_SEED = 909
    SEED = 54

    BUFFER_SIZE = 1000

    if test:
        datadir = os.path.join(path, "images/")

        dataset = tf.data.Dataset.list_files(datadir + "*.png", seed=SEED, shuffle=False)

        data_num = len([file for file in glob(str(os.path.join(datadir, "*.png")))])
        print(f"Pipeline of {data_num} images generated.")

        return dataset

    else:
        train_dir = os.path.join(path, "train/")
        val_dir = os.path.join(path, "val/")
        test_dir = os.path.join(path, "test/")

        train_dataset = tf.data.Dataset.list_files(train_dir + "images/*.png", seed=TRAIN_SEED, shuffle=False)
        val_dataset = tf.data.Dataset.list_files(val_dir + "images/*.png", seed=VAL_SEED, shuffle=False)
        test_dataset = tf.data.Dataset.list_files(test_dir + "images/*.png", seed=TEST_SEED, shuffle=False)

        if reconstruction:
            train_dataset = train_dataset.map(parse_image_reconstruction)
            val_dataset = val_dataset.map(parse_image_reconstruction)
            test_dataset = test_dataset.map(parse_image_reconstruction)
        else:
            train_dataset = train_dataset.map(parse_image)
            val_dataset = val_dataset.map(parse_image)
            test_dataset = test_dataset.map(parse_image)

        train_num = len([file for file in glob(str(os.path.join(train_dir, "images/*.png")))])
        val_num = len([file for file in glob(str(os.path.join(val_dir, "images/*.png")))])
        test_num = len([file for file in glob(str(os.path.join(test_dir, "images/*.png")))])

        dataset = {"train": train_dataset, "val": val_dataset, "test": test_dataset}

        # -- Train Dataset --#
        dataset["train"] = dataset["train"].map(load_image_train, num_parallel_calls=tf.data.AUTOTUNE)
        dataset["train"] = dataset["train"].cache()
        dataset["train"] = dataset["train"].shuffle(buffer_size=BUFFER_SIZE, seed=SEED)
        dataset["train"] = dataset["train"].repeat()
        dataset["train"] = dataset["train"].batch(bs)
        dataset["train"] = dataset["train"].prefetch(buffer_size=tf.data.AUTOTUNE)

        # -- Validation Dataset --#
        dataset["val"] = dataset["val"].map(load_image_test)
        dataset["val"] = dataset["val"].repeat()
        dataset["val"] = dataset["val"].batch(bs)
        dataset["val"] = dataset["val"].prefetch(buffer_size=tf.data.AUTOTUNE)

        # -- Test Dataset --#
        dataset["test"] = dataset["test"].map(load_image_test)
        dataset["test"] = dataset["test"].repeat()
        dataset["test"] = dataset["test"].batch(bs)
        dataset["test"] = dataset["test"].prefetch(buffer_size=tf.data.AUTOTUNE)

        print(f"{train_num} images found in {train_dir}.")
        print(f"{val_num} images found in {val_dir}.")
        print(f"{test_num} images found in {test_dir}.")

        return dataset["train"], dataset["val"], dataset["test"]



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
    parser.add_argument("--generate", "-g", default=False, help="generate slices",
                        action="store_true")
    parser.add_argument("--batch", "-b", type=int,
                        help="batch size", default=8)
    parser.add_argument("--datapath", help="path to the dataset")
    parser.add_argument("--output", help="path to the output")
    parser.add_argument("--test", "-t", help="path to the output", default=False, action="store_true")

    args = parser.parse_args()

    datapath = args.datapath
    test = args.test

    data_path = os.path.join(datapath)
    print(os.path.isdir(data_path))

    if args.generate:
        output = args.output
        generate_dataset(datapath, output, test=test)

    else:
        if args.batch:
            bs = args.batch
        else:
            bs = 32

        # some manipulation on the data
        train_set, test_set = create_pipeline_performance(datapath, bs=bs)

        for img, mask in train_set.take(1):
            print("One batch of data")
            print(img.shape, mask.shape)


if __name__ == "__main__":
    main()
import numpy as np
