# simple-classification

This project is a simple Deep learning training pipeline for classification algorithm. It should work as expected with
Tensorflow 2.5. These are the last versions at this time, newer ones might break support. Previous version as tensorflow
2.4 should work too, but we don't recommend you to use older versions.

## Installation

If you find this project on github, please follow these steps :
(Before these, make sure you already have at least tensorflow 2.4 and numpy installed)

1) Clone the repository

## Usage

The datasets for this algorithm must include separated `/training` and `/validation`
directories for training and validation datas.

For this example, I use `A Large Scale Fish Dataset` by Oğuzhan Ulucan. You can find it
at https://www.kaggle.com/crowww/a-large-scale-fish-dataset.

For the training, run the program `train.py` as explained above. You can run it directly by specifying the path to the
dataset without parameters, they will be fixed automatically. All available arguments are :

```
usage: train.py [-h] [--epochs EPOCHS] [--lr LR] [--performace]
                [--weights WEIGHTS] [--augmentation] [--batch BATCH]
                [--log LOG] [--checkpoint CHECKPOINT]
                datapath

positional arguments:
  datapath              path to the dataset

optional arguments:
  -h, --help            show this help message and exit
  --epochs EPOCHS, -e EPOCHS
                        custom epochs number
  --lr LR               custom learning rate
  --performace, -p      activate performance configuration
  --weights WEIGHTS, -w WEIGHTS
                        pretrained weights path, imagenet or None
  --augmentation, -a    activate data augmentation
  --batch BATCH, -b BATCH
                        batch size
  --log LOG             set path to logs
  --checkpoint CHECKPOINT, -c CHECKPOINT
                        set checkpoints path and name
```

These parameters can be added independently each other like :
`$ python3 train.py --epochs 50 --lr 0.0005 --performance --augmentation --batch 16 --log logs/run --checkpoint cp/model.h5 datapath=path/to/dataset`

If you use CLI method without adding parameters, epochs number will be set to 2, learning rate to 5e-4, batch size to 32
and logs path and checkpoint path to the main directory.

You can recreate this project and train it on your own dataset by :

1) Creating simple fully connected layer.
2) Add resnet50 pre-trained model.
3) Add a fully connected layer.
4) Use `model.fit` for training after compilation.

## Notes

* This repository requires Tensorflow 2.4 or higher.
* This repository is tested using Python 3.9.
