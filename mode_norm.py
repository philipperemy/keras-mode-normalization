import tensorflow as tf
import tensorflow.contrib.slim as slim


def mode_norm(inputs, _lambda, _k, _eps=1e-7):
    input_shape = tf.shape(inputs)
    print(input_shape)

    alpha = tf.Variable(1.0, name='alpha')  # for now just scalar.
    beta = tf.Variable(0.0, name='beta')  # for now just scalar.

    x = slim.flatten(inputs)
    expert_assignments = slim.fully_connected(x, num_outputs=_k, activation_fn=tf.nn.softmax, scope='mode_norm')

    xk_mean = tf.reduce_mean(tf.stack([expert_assignments[:, k:k + 1] * x for k in range(_k)]), axis=1)
    xk2_mean = tf.reduce_mean(tf.stack([expert_assignments[:, k:k + 1] * x ** 2 for k in range(_k)]), axis=1)
    variance = xk2_mean - xk_mean ** 2

    outputs = []
    for k in range(_k):
        norm_x = tf.nn.batch_normalization(x, mean=xk_mean[k], variance=variance[k],
                                           offset=beta, scale=alpha, variance_epsilon=_eps)
        outputs.append(norm_x)
    output = tf.add_n(outputs)
    output = tf.reshape(output, input_shape)
    print(output.shape)
    return output


if __name__ == '__main__':
    sess = tf.InteractiveSession()
    import numpy as np

    s = tf.constant(np.ones(shape=(64, 10, 10, 6)), dtype=tf.float32)
    b = mode_norm(s, _lambda=0.1, _k=2)
    sess.run(tf.global_variables_initializer())
    print(b.eval().shape)
