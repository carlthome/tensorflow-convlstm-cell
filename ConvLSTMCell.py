import tensorflow as tf

class ConvLSTMCell(tf.contrib.rnn.RNNCell):
  """A LSTM cell with convolutions instead of multiplications.

  Reference:
    Xingjian, S. H. I., et al. "Convolutional LSTM network: A machine learning approach for precipitation nowcasting." Advances in Neural Information Processing Systems. 2015.
  """

  def __init__(self, height, width, filters, kernel, is_training=None, new_sequences=None, statistics_timesteps=0, initializer=tf.orthogonal_initializer(), forget_bias=1.0, activation=tf.tanh):
    self._height = height
    self._width = width
    self._filters = filters
    self._kernel = kernel
    self._is_training = is_training
    self._initializer = initializer
    self._forget_bias = forget_bias
    self._activation = activation
    self._new_sequences = new_sequences
    self._statistics_timesteps = statistics_timesteps
    self._normalize = statistics_timesteps > 0

  @property
  def state_size(self):
    size = self._height * self._width * self._filters
    return tf.contrib.rnn.LSTMStateTuple(size, size)

  @property
  def output_size(self):
    return self._height * self._width * self._filters

  def __call__(self, input, state, scope=None):
    with tf.variable_scope(scope or self.__class__.__name__):
      previous_memory, previous_output = state

      with tf.variable_scope('Counter'):
        timestep = tf.Variable(0.0, trainable=False)
        increment_op = tf.cond(self._new_sequences,
                               lambda: tf.assign(timestep, 0),
                               lambda: tf.add(timestep, 1))
        with tf.control_dependencies([increment_op]):
          input = input  # TODO This assignment looks silly.

      with tf.variable_scope('Expand'):
        samples = input.get_shape()[0].value
        shape = [samples, self._height, self._width]
        input = tf.reshape(input, shape + [-1])
        previous_memory = tf.reshape(previous_memory, shape + [self._filters])
        previous_output = tf.reshape(previous_output, shape + [self._filters])

      with tf.variable_scope('Convolve'):
        channels = input.get_shape()[-1].value
        filters = self._filters
        gates = 4 * filters if filters > 1 else 4

        # The input-to-hidden and hidden-to-hidden weights can be summed directly if batch normalization is not needed.
        if not self._normalize:
          x = tf.concat([input, previous_output], axis=3)
          n = channels + filters
          m = gates
          W = tf.get_variable('Weights', self._kernel + [n, m], initializer=self._initializer)
          y = tf.nn.convolution(x, W, 'SAME')
        else:

          with tf.variable_scope('Input'):
            x = input
            n = channels
            m = gates
            W = tf.get_variable('Weights', self._kernel + [n, m], initializer=self._initializer)
            Wxh = tf.nn.convolution(x, W, 'SAME')
            Wxh = self._recurrent_batch_normalization(Wxh, timestep)

          with tf.variable_scope('Hidden'):
            x = previous_output
            n = filters
            m = gates
            W = tf.get_variable('Weights', self._kernel + [n, m], initializer=self._initializer)
            Whh = tf.nn.convolution(x, W, 'SAME')
            Whh = self._recurrent_batch_normalization(Whh, timestep)

          y = Wxh + Whh

        y += tf.get_variable('Biases', [m], initializer=tf.constant_initializer(0.0))

        input, input_gate, forget_gate, output_gate = tf.split(y, 4, axis=3)

      with tf.variable_scope('LSTM'):
        memory = (previous_memory
          * tf.sigmoid(forget_gate + self._forget_bias)
          + tf.sigmoid(input_gate) * self._activation(input))
        if self._normalize:
          memory = self._recurrent_batch_normalization(memory, timestep, offset=True)
        output = self._activation(memory) * tf.sigmoid(output_gate)

      with tf.variable_scope('Flatten'):
        shape = [-1, self._height * self._width * self._filters]
        output = tf.reshape(output, shape)
        memory = tf.reshape(memory, shape)

      return output, tf.contrib.rnn.LSTMStateTuple(memory, output)

  def _recurrent_batch_normalization(self, tensor, timestep, epsilon=1e-3, decay=0.999, offset=False):
      """Batch normalization for RNNs. Multiple population estimates are
      maintained to let the LSTM cell settle when starting a sequence.

      Notes:
        - Initial gammas should be around 0.1 to avoid vanishing gradients.
        - Statistics are calculated over time, but parameters are still shared.

      Reference:
        Cooijmans, Tim, et al. "Recurrent Batch Normalization." arXiv preprint arXiv:1603.09025 (2016).
      """
      with tf.variable_scope('Normalize'):

        # Normalize every channel/filter independently.
        filters = tensor.get_shape()[-1].value
        gamma = tf.get_variable('Scale', [filters], initializer=tf.constant_initializer(0.1))
        beta = tf.get_variable('Offset', [filters], initializer=tf.constant_initializer(0.0)) if offset else None
        batch_mean, batch_var = tf.nn.moments(tensor, [0, 1, 2])

        # TODO Vectorize.
        batch_norms = []
        for i in range(self._statistics_timesteps):
          # TODO Use tf.moving_average_variables instead.
          pop_mean = tf.get_variable('PopulationMean{}'.format(i), [filters], initializer=tf.constant_initializer(0.0), trainable=False)
          pop_var = tf.get_variable('PopulationVariance{}'.format(i), [filters], initializer=tf.constant_initializer(1.0), trainable=False)
          train_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
          train_var = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))

          def train():
            with tf.control_dependencies([train_mean, train_var]):
              return tf.nn.batch_normalization(tensor, batch_mean, batch_var, beta, gamma, epsilon)

          def test():
            return tf.nn.batch_normalization(tensor, pop_mean, pop_var, beta, gamma, epsilon)

          batch_norms.append(tf.cond(self._is_training, train, test))

        # Choose population estimate.
        idx = tf.clip_by_value(timestep, 0, self._statistics_timesteps - 1)
        predicates = [tf.equal(idx, i) for i in range(self._statistics_timesteps)]
        x = batch_norms[-1]
        for i in range(self._statistics_timesteps):
          x = tf.cond(predicates[i], lambda: batch_norms[i], lambda: x)
        return x
        """TODO Use tf.case instead when fixed: https://github.com/tensorflow/tensorflow/issues/3334
        return tf.case(list(zip(predicates, batch_norms)),
                       default=batch_norms[-1],
                       exclusive=True)
        """


def flatten(tensor):
  samples, timesteps, height, width, filters = tensor.get_shape().as_list()
  return tf.reshape(tensor, [samples, timesteps, height * width * filters])


def expand(tensor, height, width):
  samples, timesteps, features = tensor.get_shape().as_list()
  return tf.reshape(tensor, [samples, timesteps, height, width, -1])
