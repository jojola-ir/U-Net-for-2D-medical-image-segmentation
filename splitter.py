'''Dataset splitter'''
import argparse

import splitfolders


def random_splitter(src, dest, test_rate):
    """Splits generated images (slices) into train/val/test subdirectories.

        Parameters
        ----------
        src : str
            Path to the slices directory.

        dest : str
            Path to the destination directory.

        test_rate : float
            Test images rate.
        """

    assert test_rate < 1, "test_rate must be less than 1"

    splitfolders.ratio(src, output=dest,
                       seed=1337, ratio=(1 - test_rate, 0, test_rate), group_prefix=None, move=False)



def main():
    """This is the main function"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", help="path to the dataset")
    parser.add_argument("--output", "-o", help="path to the output")
    parser.add_argument("--testrate", "-t", help="part of test set", default=0.2)

    args = parser.parse_args()

    datapath = args.datapath
    output = args.output
    test_rate = args.testrate

    random_splitter(datapath, output, test_rate)



if __name__ == "__main__":
    main()
