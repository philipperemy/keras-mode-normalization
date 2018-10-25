'''Trains a simple convnet on the MNIST dataset.
Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function

import keras
import numpy as np
from argparse import ArgumentParser
from keras.datasets import mnist, cifar10, fashion_mnist
from keras.layers import Conv2D, BatchNormalization, AveragePooling2D
from keras.layers import Dense, Flatten
from keras.models import Sequential

import svhn
from mode_normalization import ModeNormalization


def arg_parse():
    arg_p = ArgumentParser()
    arg_p.add_argument('--mode_norm', action='store_true')
    arg_p.add_argument('--batch_norm', action='store_true')
    return arg_p


args = arg_parse().parse_args()

batch_size = 128
num_classes = 40
epochs = 15

# input image dimensions
img_rows, img_cols = 28, 28


def data_preprocessing(x_train, y_train, x_test, y_test):
    # if K.image_data_format() == 'channels_first':
    #     x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    #     x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    #     input_shape = (1, img_rows, img_cols)
    # else:
    #     x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    #     x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    #     input_shape = (img_rows, img_cols, 1)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    return (x_train, y_train), (x_test, y_test)


def get_data(func):
    (x_train, y_train), (x_test, y_test) = func()
    return data_preprocessing(x_train, y_train, x_test, y_test)


(x_tr_mnist, y_tr_mnist), (x_te_mnist, y_te_mnist) = get_data(mnist.load_data)
(x_tr_cifar10, y_tr_cifar10), (x_te_cifar10, y_te_cifar10) = get_data(cifar10.load_data)
(x_tr_fmnist, y_tr_fmnist), (x_te_fmnist, y_te_fmnist) = get_data(fashion_mnist.load_data)
(x_tr_svhn, y_tr_svhn), (x_te_svhn, y_te_svhn) = get_data(svhn.load_data)

x_tr_svhn = np.mean(x_tr_svhn[:, :28, :28], axis=-1)
x_te_svhn = np.mean(x_te_svhn[:, :28, :28], axis=-1)

x_tr_cifar10 = np.mean(x_tr_cifar10[:, :28, :28], axis=-1)
x_te_cifar10 = np.mean(x_te_cifar10[:, :28, :28], axis=-1)

y_tr_svhn = y_tr_svhn.squeeze()
y_te_svhn = y_te_svhn.squeeze()

y_tr_cifar10 = y_tr_cifar10.squeeze()
y_te_cifar10 = y_te_cifar10.squeeze()

y_tr_fmnist = y_tr_fmnist + 20
y_te_fmnist = y_te_fmnist + 20

y_tr_cifar10 = y_tr_cifar10 + 10
y_te_cifar10 = y_te_cifar10 + 10

y_tr_svhn = y_tr_svhn + 29
y_te_svhn = y_te_svhn + 29

x_train = np.concatenate((x_tr_mnist, x_tr_cifar10, x_tr_fmnist, x_tr_svhn), axis=0)
y_train = np.concatenate((y_tr_mnist, y_tr_cifar10, y_tr_fmnist, y_tr_svhn), axis=0)

x_test = np.concatenate((x_te_mnist, x_te_cifar10, x_te_fmnist, x_te_svhn), axis=0)
y_test = np.concatenate((y_te_mnist, y_te_cifar10, y_te_fmnist, y_te_svhn), axis=0)

x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

# add a mode to the data.
# train_indices = np.random.choice(a=range(len(x_train)), size=len(x_train) // 2, replace=False)
# x_train[train_indices] = 1.0 - x_train[train_indices]
#
# test_indices = np.random.choice(a=range(len(x_test)), size=len(x_test) // 2, replace=False)
# x_test[test_indices] = 1.0 - x_test[test_indices]

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes=num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes=num_classes)


def add_norm_layer(m):
    if args.batch_norm:
        m.add(BatchNormalization(momentum=0.9))
    if args.mode_norm:
        m.add(ModeNormalization(k=2, momentum=0.9))


# LeNet-5 https://github.com/olramde/LeNet-keras
model = Sequential()
model.add(
    Conv2D(6, kernel_size=(5, 5), strides=(1, 1), activation='tanh', input_shape=x_train.shape[1:], padding="same"))
model.add(AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding='valid'))
add_norm_layer(model)
model.add(Conv2D(16, kernel_size=(5, 5), strides=(1, 1), activation='tanh', padding='valid'))
model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
add_norm_layer(model)
model.add(Conv2D(120, kernel_size=(5, 5), strides=(1, 1), activation='tanh', padding='valid'))
model.add(Flatten())
model.add(Dense(84, activation='tanh'))
model.add(Dense(num_classes, activation='softmax'))

# model.add(Conv2D(64, kernel_size=(3, 3),
#                  activation='relu',
#                  input_shape=x_train.shape[1:]))
# add_norm_layer(model)
# model.add(Conv2D(128, (3, 3), activation='relu'))
# add_norm_layer(model)
# model.add(MaxPooling2D(pool_size=(2, 2)))
# # model.add(Dropout(0.5))
# model.add(Flatten())
# model.add(Dense(256, activation='relu'))
# # add_norm_layer(model) # TODO does not work here!
# # model.add(Dropout(0.5))
# model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(lr=0.001),
              metrics=['accuracy'])

model.summary()

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
