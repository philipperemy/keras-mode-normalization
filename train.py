import os
import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
from glob import glob
import config as cfg
from lenet import Lenet


def main():
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    sess = tf.Session()
    batch_size = cfg.BATCH_SIZE
    save_path = cfg.PARAMETER_FILE
    le_net = Lenet()
    max_iter = cfg.MAX_ITER

    saver = tf.train.Saver()
    if len(glob(save_path + '*')) > 0:
        print(f'Restoring from {save_path}.')
        saver.restore(sess, save_path)
    else:
        sess.run(tf.global_variables_initializer())
        print(f'Fresh initialization.')

    for i in range(max_iter):
        batch = mnist.train.next_batch(batch_size)
        tr_feed_dict = {le_net.raw_input_image: batch[0], le_net.raw_input_label: batch[1]}
        if i % 100 == 0:
            train_accuracy = sess.run(le_net.train_accuracy, feed_dict=tr_feed_dict)
            print(f'step {str(i).ljust(6)} training accuracy {train_accuracy:.3f}')
            saver.save(sess, save_path)
        sess.run(le_net.train_op, feed_dict=tr_feed_dict)


if __name__ == '__main__':
    main()
