'''Training'''

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from skimage import measure
from tensorflow import keras

from data import create_pipeline_performance
from losses import weighted_cross_entropy
from metrics import dice_coeff


def display(display_list):
    plt.figure(figsize=(15, 15))

    title = ['Input Image', 'Predicted Mask', 'Segmented image']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        if title[i] == "Segmented image":
            display_list[i] = np.array(display_list[i])
            msk = display_list[i-1][:,:,0]
            contours = measure.find_contours(msk, 0.8)
            plt.imshow(display_list[i], cmap='gray')
            for contour in contours:
                plt.plot(contour[:, 1], contour[:, 0], linewidth=2)
        else:
            plt.imshow(keras.utils.array_to_img(display_list[i]), cmap='gray')
        plt.axis('off')
    plt.show()

def show_predictions(dataset, model, num=1):
    for image, _ in dataset.take(num):
        pred_mask = (model.predict(image)).astype("uint8")
        segmented = image
        rd = np.random.randint(pred_mask.shape[0])
        display([image[rd], pred_mask[rd], segmented[rd]])


def main():
    """Main function."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--modelpath", default="/models/run1.h5",
                        help="path to .h5 file for inference")
    parser.add_argument("--datapath", help="path to the dataset")
    parser.add_argument("--num", "-n", default=4, help="number of samples to take for prediction")

    args = parser.parse_args()

    datapath = args.datapath
    model_path = args.modelpath
    num = int(args.num)

    # data loading
    path = os.path.join(datapath)
    _, _, test_data = create_pipeline_performance(path=path, test=False)

    # model loading
    model = keras.models.load_model(model_path,
                                    custom_objects={'weighted_cross_entropy': weighted_cross_entropy,
                                                    'dice_coeff': dice_coeff})

    model.summary()

    show_predictions(dataset=test_data, model=model, num=num)


if __name__ == "__main__":
    main()
