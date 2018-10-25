# https://github.com/aditya9211/SVHN-CNN
import numpy as np
import scipy.io as sio


def load_data():
    """Loads the SVHN dataset.

    # Returns
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    """
    train = sio.loadmat('svhn/train_32x32.mat')
    test = sio.loadmat('svhn/test_32x32.mat')

    x_train, y_train = train['X'], train['y']
    x_test, y_test = test['X'], test['y']

    x_train = np.transpose(x_train, (3, 0, 1, 2))
    x_test = np.transpose(x_test, (3, 0, 1, 2))

    return (x_train, y_train), (x_test, y_test)
