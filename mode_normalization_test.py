from __future__ import print_function

import unittest

import numpy as np
from keras import Input, Model
from keras.layers import BatchNormalization, Dense, Flatten
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical

from mode_normalization import ModeNormalization


class ExecutionTest(unittest.TestCase):

    def test_1(self):
        # ModeNormalization with only one mode is equivalent to BatchNormalization
        a = np.random.uniform(size=(50, 10, 10, 3))

        i1 = Input(shape=(10, 10, 3))
        x1 = ModeNormalization(k=1)(i1)
        m1 = Model(inputs=[i1], outputs=[x1])
        p1 = m1.predict(a)
        print(p1.shape)

        i2 = Input(shape=(10, 10, 3))
        x2 = BatchNormalization()(i2)
        m2 = Model(inputs=[i2], outputs=[x2])
        p2 = m2.predict(a)
        print(p2.shape)

        np.testing.assert_almost_equal(p1, p2)

    def test_2(self):
        num_modes = 6
        input_shape = (50, 10, 10, 3)
        a = np.random.uniform(size=input_shape)

        i1 = Input(shape=(10, 10, 3))
        x1 = ModeNormalization(k=num_modes)(i1)
        m1 = Model(inputs=[i1], outputs=[x1])
        p1 = m1.predict(a)
        assert input_shape == p1.shape

    def test_3(self):
        num_modes = 6
        h, w, num_channels = 10, 10, 3
        input_shape = (50, h, w, num_channels)
        a = np.random.uniform(size=input_shape)

        i1 = Input(shape=(h, w, num_channels))
        x1 = ModeNormalization(k=num_modes)(i1)
        m1 = Model(inputs=[i1], outputs=[x1])

        weight_shapes = [a.shape for a in m1.get_weights()]
        assert weight_shapes == [(num_channels, num_modes),  # gates_kernel
                                 (num_modes,),  # gates_bias
                                 (num_channels,),  # gates_gamma
                                 (num_channels,),  # gates_beta
                                 (num_modes, num_channels),  # moving_mean
                                 (num_modes, num_channels)]  # moving_variance

    def test_4(self):
        num_modes = 3
        h, w, num_channels = 10, 10, 3
        input_shape = (50, h, w, num_channels)
        a = np.random.uniform(size=input_shape)
        b = np.random.uniform(size=input_shape)

        i1 = Input(shape=(h, w, num_channels))
        x1 = ModeNormalization(k=num_modes)(i1)
        m1 = Model(inputs=[i1], outputs=[x1])
        m1.compile(optimizer='adam', loss='mse')
        p1 = m1.predict(b)
        m1.fit(a, a, epochs=2)
        p2 = m1.predict(b)
        np.testing.assert_equal(np.any(np.not_equal(p1, p2)), True)

    def test_5(self):
        h, w, num_channels = 10, 10, 3
        input_shape = (50, h, w, num_channels)
        a = np.random.uniform(size=input_shape)
        b = np.random.uniform(size=input_shape)

        i1 = Input(shape=(h, w, num_channels))
        x1 = ModeNormalization(k=1)(i1)
        m1 = Model(inputs=[i1], outputs=[x1])
        m1.compile(optimizer='adam', loss='mse')
        m1.fit(a, a, epochs=1)
        p1 = m1.predict(b)

        i2 = Input(shape=(h, w, num_channels))
        x2 = BatchNormalization()(i2)
        m2 = Model(inputs=[i2], outputs=[x2])
        m2.compile(optimizer='adam', loss='mse')
        m2.fit(a, a, epochs=1)
        p2 = m2.predict(b)

        np.testing.assert_almost_equal(p1, p2, decimal=1)

    def test_6(self):
        h, w, num_channels = 10, 10, 3
        input_shape = (50, h, w, num_channels)
        mode1 = np.random.uniform(size=input_shape, low=0, high=1)
        mode2 = np.random.uniform(size=input_shape, low=-1, high=0)
        x_data = np.vstack([mode1, mode2])
        y_data = to_categorical(np.vstack([0] * 50 + [1] * 50), num_classes=2)

        i1 = Input(shape=(h, w, num_channels))
        x1 = ModeNormalization(k=2)(i1)
        x1 = Flatten()(x1)
        x1 = Dense(2, activation='softmax')(x1)
        m1 = Model(inputs=[i1], outputs=[x1])
        m1.compile(optimizer=Adam(lr=0.01), loss='categorical_crossentropy')
        m1.fit(x_data, y_data, epochs=10, shuffle=True)

        def gate_inference(x):
            return (np.dot(np.mean(x, axis=(1, 2)), m1.get_weights()[0]) + m1.get_weights()[1]).argmax(axis=-1)

        mode1_val = np.mean(gate_inference(mode1))
        mode2_val = np.mean(gate_inference(mode2))

        # It's possible that in some cases, the network cannot really separate the two modes.
        # I would say it fails ~5% of the time.
        if mode1_val < mode2_val:
            assert mode1_val < 0.3 and mode2_val >= 0.7
        else:
            assert mode2_val < 0.3 and mode1_val >= 0.7


if __name__ == '__main__':
    ExecutionTest().test_6()
