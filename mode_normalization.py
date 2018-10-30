import keras.backend as K
from keras import initializers, regularizers, constraints
from keras.engine import Layer, InputSpec
from keras.layers import activations
from keras.legacy import interfaces


class ModeNormalization(Layer):
    """Mode Normalization layer.

    Normalize the activations of the previous layer at each batch,
    i.e. applies a transformation that maintains the mean activation
    close to 0 and the activation standard deviation close to 1 for K
    different modes.

    # Arguments
        axis: Integer, the axis that should be normalized
            (typically the features axis).
            For instance, after a `Conv2D` layer with
            `data_format="channels_first"`,
            set `axis=1` in `ModeNormalization`.
        k: Integer, the number of modes of the normalization.
        momentum: Momentum for the moving mean and the moving variance.
        epsilon: Small float added to variance to avoid dividing by zero.
        center: If True, add offset of `beta` to normalized tensor.
            If False, `beta` is ignored.
        scale: If True, multiply by `gamma`.
            If False, `gamma` is not used.
            When the next layer is linear (also e.g. `nn.relu`),
            this can be disabled since the scaling
            will be done by the next layer.
        beta_initializer: Initializer for the beta weight.
        gamma_initializer: Initializer for the gamma weight.
        moving_mean_initializer: Initializer for the moving mean.
        moving_variance_initializer: Initializer for the moving variance.
        beta_regularizer: Optional regularizer for the beta weight.
        gamma_regularizer: Optional regularizer for the gamma weight.
        beta_constraint: Optional constraint for the beta weight.
        gamma_constraint: Optional constraint for the gamma weight.

    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.

    # Output shape
        Same shape as input.

    # References
        - [Mode Normalization] https://arxiv.org/pdf/1810.05466v1.pdf
    """

    @interfaces.legacy_batchnorm_support
    def __init__(self,
                 axis=-1,
                 k=2,
                 momentum=0.99,
                 epsilon=1e-3,
                 center=True,
                 scale=True,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 moving_mean_initializer='zeros',
                 moving_variance_initializer='ones',
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 **kwargs):
        super(ModeNormalization, self).__init__(**kwargs)
        self.supports_masking = True
        self.axis = axis
        self.k = k
        self.momentum = momentum
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.moving_mean_initializer = initializers.get(moving_mean_initializer)
        self.moving_variance_initializer = (
            initializers.get(moving_variance_initializer))
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_constraint = constraints.get(gamma_constraint)

    def build(self, input_shape):
        dim = input_shape[self.axis]
        if dim is None:
            raise ValueError('Axis ' + str(self.axis) + ' of '
                                                        'input tensor should have a defined dimension '
                                                        'but the layer received an input with shape ' +
                             str(input_shape) + '.')
        self.input_spec = InputSpec(ndim=len(input_shape),
                                    axes={self.axis: dim})

        shape = (dim,)
        moving_shape = (self.k, dim,)

        self.gates_kernel = self.add_weight(shape=(dim, self.k),
                                            initializer=initializers.get('glorot_uniform'),
                                            name='gates_kernel',
                                            regularizer=None,
                                            constraint=None)

        self.gates_bias = self.add_weight(shape=(self.k,),
                                          initializer=initializers.get('zeros'),
                                          name='gates_bias')

        if self.scale:
            self.gamma = self.add_weight(shape=shape,
                                         name='gates_gamma',
                                         initializer=self.gamma_initializer,
                                         regularizer=self.gamma_regularizer,
                                         constraint=self.gamma_constraint)
        else:
            self.gamma = None
        if self.center:
            self.beta = self.add_weight(shape=shape,
                                        name='gates_beta',
                                        initializer=self.beta_initializer,
                                        regularizer=self.beta_regularizer,
                                        constraint=self.beta_constraint)
        else:
            self.beta = None
        self.moving_mean = self.add_weight(
            shape=moving_shape,
            name='moving_mean',
            initializer=self.moving_mean_initializer,
            trainable=False)
        self.moving_variance = self.add_weight(
            shape=moving_shape,
            name='moving_variance',
            initializer=self.moving_variance_initializer,
            trainable=False)
        self.built = True

    def apply_gates(self, inputs, input_shape, axis):
        gates = K.dot(K.mean(inputs, axis=axis), self.gates_kernel)
        gates = K.bias_add(gates, self.gates_bias, data_format='channels_last')
        gates = activations.get('softmax')(gates)
        inputs_mul_gates = K.stack([K.reshape(gates[:, k], [-1] + [1] * (len(input_shape) - 1)) * inputs
                                    for k in range(self.k)], axis=0)
        return inputs_mul_gates

    def call(self, inputs, training=None):
        input_shape = K.int_shape(inputs)
        # Prepare broadcasting shape.
        ndim = len(input_shape)
        reduction_axes = list(range(len(input_shape)))
        del reduction_axes[self.axis]
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[self.axis] = input_shape[self.axis]
        needs_broadcasting = (sorted(reduction_axes) != list(range(ndim))[:-1])

        def normalize_inference():

            def apply_mode_normalization_inference(moving_mean, moving_variance, beta, gamma):
                inputs_mul_gates_ = self.apply_gates(inputs, input_shape, reduction_axes[1:])
                outputs = []
                for k_ in range(self.k):
                    outputs.append(K.batch_normalization(
                        inputs_mul_gates_[k_],
                        moving_mean[k_],
                        moving_variance[k_],
                        beta / self.k,
                        gamma,
                        axis=self.axis,
                        epsilon=self.epsilon))
                return K.sum(K.stack(outputs, axis=0), axis=0)

            if needs_broadcasting:
                # In this case we must explicitly broadcast all parameters.
                broadcast_moving_mean = K.reshape(self.moving_mean,
                                                  broadcast_shape)
                broadcast_moving_variance = K.reshape(self.moving_variance,
                                                      broadcast_shape)
                if self.center:
                    broadcast_beta = K.reshape(self.beta, broadcast_shape)
                else:
                    broadcast_beta = None
                if self.scale:
                    broadcast_gamma = K.reshape(self.gamma,
                                                broadcast_shape)
                else:
                    broadcast_gamma = None
                return apply_mode_normalization_inference(broadcast_moving_mean, broadcast_moving_variance,
                                                          broadcast_beta, broadcast_gamma)
            else:
                return apply_mode_normalization_inference(self.moving_mean, self.moving_variance,
                                                          self.beta, self.gamma)

        # If the learning phase is *static* and set to inference:
        if training in {0, False}:
            return normalize_inference()

        inputs_mul_gates = self.apply_gates(inputs, input_shape, reduction_axes[1:])

        # training.
        mean_list, variance_list, normed_training_list = [], [], []
        norm_func = K.normalize_batch_in_training
        for k in range(self.k):
            normed_training, mean, variance = norm_func(inputs_mul_gates[k], self.gamma, self.beta / self.k,
                                                        reduction_axes, epsilon=self.epsilon)
            normed_training_list.append(normed_training)
            mean_list.append(mean)
            variance_list.append(variance)

        mean = K.stack(mean_list, axis=0)
        variance = K.stack(variance_list, axis=0)
        normed_training = K.sum(normed_training_list, axis=0)

        if K.backend() != 'cntk':
            sample_size = K.prod([K.shape(inputs)[axis]
                                  for axis in reduction_axes])
            sample_size = K.cast(sample_size, dtype=K.dtype(inputs))

            # sample variance - unbiased estimator of population variance
            variance *= sample_size / (sample_size - (1.0 + self.epsilon))

        self.add_update([K.moving_average_update(self.moving_mean,
                                                 mean,
                                                 self.momentum),
                         K.moving_average_update(self.moving_variance,
                                                 variance,
                                                 self.momentum)],
                        inputs)

        # Pick the normalized form corresponding to the training phase.
        return K.in_train_phase(normed_training,
                                normalize_inference,
                                training=training)

    def get_config(self):
        config = {
            'axis': self.axis,
            'k': self.k,
            'momentum': self.momentum,
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale,
            'beta_initializer': initializers.serialize(self.beta_initializer),
            'gamma_initializer': initializers.serialize(self.gamma_initializer),
            'moving_mean_initializer':
                initializers.serialize(self.moving_mean_initializer),
            'moving_variance_initializer':
                initializers.serialize(self.moving_variance_initializer),
            'beta_regularizer': regularizers.serialize(self.beta_regularizer),
            'gamma_regularizer': regularizers.serialize(self.gamma_regularizer),
            'beta_constraint': constraints.serialize(self.beta_constraint),
            'gamma_constraint': constraints.serialize(self.gamma_constraint)
        }
        base_config = super(ModeNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape
