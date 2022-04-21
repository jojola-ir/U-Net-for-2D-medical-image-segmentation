'''Dataset splitter'''
import argparse
import os

import splitfolders


def random_splitter(src, dest, test_rate, clear):
    """Splits generated images (slices) into train/val/test subdirectories.

        Parameters
        ----------
        src : str
            Path to the slices directory.

        dest : str
            Path to the destination directory.

        test_rate : float
            Test images rate.

        clear : boolean
            Enables or disables useless images clearing.
        """

    assert test_rate < 1, "test_rate must be less than 1"

    if clear:
        k = 0
        for root, _, files in os.walk(src):
            if root == os.path.join(src, "masks"):
                for f in files:
                    if not f.endswith(".DS_Store"):
                        maskpath = os.path.join(root, f)
                        imagepath = os.path.join(os.path.join(src, "images/"), f)
                        if os.lstat(maskpath).st_size <= 120:  # set file size in kb
                            k += 1
                            print(f)
                            os.remove(imagepath)
                            os.remove(maskpath)
        print(f"Images found : {k}")

    val_rate = 0.1

    splitfolders.ratio(src, output=dest,
                       seed=1337, ratio=(1 - (test_rate + val_rate), val_rate, test_rate), group_prefix=None, move=False)



def main():
    """This is the main function"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", help="path to the dataset")
    parser.add_argument("--output", "-o", help="path to the output")
    parser.add_argument("--testrate", "-t", help="part of test set", default=0.2)
    parser.add_argument("--clear", "-c", help="clear useless images", default=False, action="store_true")

    args = parser.parse_args()

    datapath = args.datapath
    output = args.output
    test_rate = args.testrate
    clear = args.clear

    random_splitter(datapath, output, test_rate, clear)



if __name__ == "__main__":
    main()
