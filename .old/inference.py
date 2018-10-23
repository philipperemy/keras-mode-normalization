import numpy as np
import tensorflow as tf
from PIL import Image

import config as cfg
from lenet import Lenet


class Inference:
    def __init__(self):
        self.le_net = Lenet()
        self.sess = tf.Session()
        self.parameter_path = cfg.PARAMETER_FILE
        self.saver = tf.train.Saver()

    def predict(self, image):
        img = image.convert('L')
        img = img.resize([28, 28], Image.ANTIALIAS)
        image_input = np.array(img, dtype="float32") / 255
        image_input = np.reshape(image_input, [-1, 784])

        self.saver.restore(self.sess, self.parameter_path)
        prediction = self.sess.run(self.le_net.prediction, feed_dict={self.le_net.raw_input_image: image_input})
        return prediction
