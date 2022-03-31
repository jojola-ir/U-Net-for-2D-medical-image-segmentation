'''Training'''

import argparse
import os

from tensorflow import keras

from data import create_pipeline
from model import custom_model, unet

NUM_TRAIN = 10240
NUM_TEST = 2560


def model_builder(model, datapath, pw, da):
    """Build a model by calling custom_model function.

    The number of output neurons is automatically set to the number of classes
    that make up the dataset.

    Parameters
    ----------
    datapath : str
        Path to dataset directory

    da: boolean
        Enables or disable data augmentation

    Returns
    -------
    model
        builded model
    """
    # # system config: seed
    # keras.backend.clear_session()
    # tf.random.set_seed(42)
    # np.random.seed(42)

    if model == "resnet50":
        training_dir = os.path.join(datapath, "training")
        classes = [name for name in os.listdir(training_dir) if os.path.isdir(
            os.path.join(training_dir, name))]

        clNbr = len(classes)
        model = custom_model(clNbr, pw, da)

    elif model == "unet":
        model = unet(4)

    return model


def create_callbacks(run_logdir, checkpoint_path="model.h5", early_stop=False):
    """Creates a tab composed of defined callbacks.

    Early stopping is disabled by default.

    All checkpoints saved by tensorboard will be stored in a new directory
    named /logs in main folder.
    The final .h5 file will also be stored in a new directory named /models.

    Parameters
    ----------
    run_logdir : str
        Path to logs directory, create a new one if it doesn't exist.

    checkpoint_path : str
        Path to model directory, create a new one if it doesn't exist.

    early_stop : boolean (False by default)
        Enables or disables early stopping.

    Returns
    -------
    list
        a list of defined callbacks
    """

    callbacks = []

    if early_stop:
        early_stopping_cb = keras.callbacks.EarlyStopping(patience=20)
        callbacks.append(early_stopping_cb)

    checkpoint_cb = keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                    save_best_only=True)
    callbacks.append(checkpoint_cb)

    tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)
    callbacks.append(tensorboard_cb)

    return callbacks


def main():
    """Main function."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", "-e", type=int, default=10,
                        help="custom epochs number")
    parser.add_argument("--lr", type=float, default=5e-4,
                        help="custom learning rate")
    parser.add_argument("--performace", "-p", default=False,
                        help="activate performance configuration",
                        action="store_true")
    parser.add_argument("--weights", "-w", default="imagenet",
                        help="pretrained weights path, imagenet or None")
    parser.add_argument("--augmentation", "-a", default=False,
                        help="activate data augmentation",
                        action="store_true")
    parser.add_argument("--batch", "-b", type=int, default=32,
                        help="batch size")
    parser.add_argument("--datapath", help="path to the dataset")
    parser.add_argument("--log", default="logs/run2", help="set path to logs")
    parser.add_argument("--checkpoint", "-c", default="models/run1.h5",
                        help="set checkpoints path and name")

    args = parser.parse_args()

    datapath = args.datapath
    epochs = args.epochs
    lr = args.lr
    logpath = args.log
    cppath = args.checkpoint
    performance = args.performace
    pretrained_weights = args.weights
    da = args.augmentation
    bs = args.batch

    # data loading
    path = os.path.join(datapath)
    # test_path = os.path.join(datapath, "test/")
    train_set, val_set, test_set = create_pipeline(path, bs=bs)

    # model building
    model = model_builder("unet", datapath, pretrained_weights, da)
    optimizer = keras.optimizers.Nadam(learning_rate=lr)
    model.compile(loss="binary_crossentropy",
                  optimizer=optimizer,
                  metrics=["accuracy"])

    model.summary()
    # callbacks
    run_logs = logpath
    checkpoint_path = cppath
    cb = create_callbacks(run_logs, checkpoint_path)

    EPOCH_STEP_TRAIN = NUM_TRAIN // bs
    EPOCH_STEP_TEST = NUM_TEST // bs

    # training and evaluation
    model.fit_generator(generator=train_set,
                        steps_per_epoch=EPOCH_STEP_TRAIN,
                        validation_data=val_set,
                        validation_steps=EPOCH_STEP_TEST,
                        epochs=epochs,
                        callbacks=[cb])

    _, test_metrics = model.evaluate(test_set)
    print("Test set accuracy: {:.02f}%".format(test_metrics * 100))


if __name__ == "__main__":
    main()
