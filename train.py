'''Training'''

import argparse
import os
from datetime import datetime

import segmentation_models as sm
from tensorflow import keras

from data import create_pipeline, create_pipeline_performance
from losses import wce_dice_loss
from metrics import dice_coeff
from model import custom_model, multi_task_unet

NUM_TRAIN = 2720
NUM_TEST = 850

sm.set_framework('tf.keras')
sm.framework()


def model_builder(model, datapath, pw, da=False, reconstruction=True, segmentation=False, custom_weights=None):
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

    else:
        model = multi_task_unet(5, reconstruction=reconstruction, segmentation=segmentation, custom_weights=custom_weights)

    return model


def create_callbacks(run_logdir, checkpoint_path="model.h5", patience=2, early_stop=False):
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
        print(f"Early stopping patience : {patience}")
        early_stopping_cb = keras.callbacks.EarlyStopping(monitor="val_dice_coeff",
                                                          patience=patience,
                                                          mode="max",
                                                          verbose=1)
        callbacks.append(early_stopping_cb)

    checkpoint_cb = keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                    save_best_only=True)
    callbacks.append(checkpoint_cb)

    tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)
    callbacks.append(tensorboard_cb)

    remote_monitor_cb = keras.callbacks.RemoteMonitor(root="http://localhost:9000",
                                                      path="/publish/epoch/end/",
                                                      field="data")
    callbacks.append(remote_monitor_cb)

    return callbacks


def main():
    """Main function."""
    now = datetime.now()

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", "-e", type=int, default=10,
                        help="custom epochs number")
    parser.add_argument("--lr", type=float, default=5e-4,
                        help="custom learning rate")
    parser.add_argument("--performace", "-p", default=False,
                        help="activate performance configuration",
                        action="store_true")
    parser.add_argument("--weights", "-w", default=None,
                        help="pretrained weights path, imagenet or None")
    parser.add_argument("--custom_model", default="unet",
                        help="load custom or combined model")
    parser.add_argument("--multitask", default=False,
                        help="activate reconstruction and segmentation",
                        action="store_true")
    parser.add_argument("--reconstruction", "-r", default=False,
                        help="train model to reconstruct input image",
                        action="store_true")
    parser.add_argument("--segmentation", "-s", default=False,
                        help="train model to segment input image",
                        action="store_true")
    parser.add_argument("--load", default=False,
                        help="load previous model",
                        action="store_true")
    parser.add_argument("--modelpath", default="/models/run1.h5",
                        help="path to .h5 file for transfert learning")
    parser.add_argument("--batch", "-b", type=int, default=16,
                        help="batch size")
    parser.add_argument("--datapath", help="path to the dataset")
    parser.add_argument("--log", default=f"logs/run{now.strftime('%m_%d_%H_%M')}", help="set path to logs")
    parser.add_argument("--checkpoint", "-c", default=f"models/benchmarks/run{now.strftime('%m_%d_%H_%M')}.h5",
                        help="set checkpoints path and name")

    args = parser.parse_args()

    datapath = args.datapath
    custom_model = args.custom_model
    reconstruction = args.reconstruction
    segmentation = args.segmentation
    weights = args.weights
    load_model = args.load
    epochs = args.epochs
    lr = args.lr
    logpath = args.log
    cppath = args.checkpoint
    performance = args.performace
    pretrained_weights = args.weights
    bs = args.batch

    # data loading
    path = os.path.join(datapath)

    if performance:
        train_set, val_set, test_set = create_pipeline_performance(path, reconstruction=reconstruction, bs=bs)
    else:
        train_set, val_set, test_set = create_pipeline(path, bs=bs)


    dice_loss = sm.losses.DiceLoss()
    bf_loss = sm.losses.BinaryFocalLoss()
    wce_dice = wce_dice_loss

    # model building
    if load_model:
        model_name = "custom"
        model_path = args.modelpath
        model = keras.models.load_model(model_path, compile=False)
        print(f"Transfert learning from {model_path}")
    else:
        model_name = custom_model

        if weights != None:
            weights = keras.models.load_model(weights, compile=False)
        model = model_builder(model=model_name, datapath=datapath, pw=pretrained_weights,
                              reconstruction=reconstruction, segmentation=segmentation, custom_weights=weights)

    losses = [wce_dice]

    metrics = [dice_coeff]

    if reconstruction:
        optimizer = keras.optimizers.Adam(learning_rate=lr)
        model.compile(loss="mean_squared_error",
                      optimizer=optimizer,
                      metrics=metrics)
    else:
        optimizer = keras.optimizers.Nadam(learning_rate=lr)
        model.compile(loss=losses,
                      optimizer=optimizer,
                      metrics=metrics)

    model.summary()

    model_architecture_path = "architecture/"
    if os.path.exists(model_architecture_path) is False:
        os.makedirs(model_architecture_path)

    keras.utils.plot_model(model,
                           to_file=os.path.join(model_architecture_path,
                                                f"model_unet_{model_name}_{now.strftime('%m_%d_%H_%M')}.png"),
                           show_shapes=True)

    # callbacks
    run_logs = logpath
    checkpoint_path = cppath
    if epochs < 40:
        cb_patience = 5
    else:
        cb_patience = epochs // 10
    cb = create_callbacks(run_logs, checkpoint_path, cb_patience, True)

    EPOCH_STEP_TRAIN = NUM_TRAIN // bs
    EPOCH_STEP_TEST = NUM_TEST // bs

    # training and evaluation
    """model.fit_generator(generator=train_set,
                        steps_per_epoch=EPOCH_STEP_TRAIN,
                        validation_data=val_set,
                        validation_steps=EPOCH_STEP_TEST,
                        epochs=epochs,
                        callbacks=[cb])"""

    model.fit(x=train_set,
              epochs=epochs,
              verbose="auto",
              callbacks=cb,
              validation_data=val_set,
              steps_per_epoch=EPOCH_STEP_TRAIN,
              validation_steps=EPOCH_STEP_TEST)

    if reconstruction:
        _, mae_metrics, dice_metrics = model.evaluate(x=test_set,
                                     steps=EPOCH_STEP_TEST)
        print("MAE : {:.02f}".format(mae_metrics))
        print("Dice : {:.02f}".format(dice_metrics))
    else:
        _, dice_metrics = model.evaluate(x=test_set,
                                                       steps=EPOCH_STEP_TEST)
        print("Dice coefficient : {:.02f}".format(dice_metrics))


if __name__ == "__main__":
    main()
