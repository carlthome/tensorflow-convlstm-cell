import tensorflow as tf

class ConvLSTMCell(tf.contrib.rnn.RNNCell):
  """A LSTM cell with convolutions instead of multiplications.

  Reference:
    Xingjian, S. H. I., et al. "Convolutional LSTM network: A machine learning approach for precipitation nowcasting." Advances in Neural Information Processing Systems. 2015.
  """

  def __init__(self, height, width, filters, kernel, initializer=None, forget_bias=1.0, activation=tf.tanh):
    self._height = height
    self._width = width
    self._filters = filters
    self._kernel = kernel
    self._initializer = initializer
    self._forget_bias = forget_bias
    self._activation = activation
    self._size = int(self._height * self._width * self._filters)

  @property
  def state_size(self):
    return tf.contrib.rnn.LSTMStateTuple(self._size, self._size)

  @property
  def output_size(self):
    return self._size

  def __call__(self, input, state, scope=None):
    with tf.variable_scope(scope or self.__class__.__name__):
      previous_memory, previous_output = state

      with tf.variable_scope('Expand'):
        samples = input.get_shape()[0].value
        shape = [samples, self._height, self._width, -1]
        input = tf.reshape(input, shape)
        previous_memory = tf.reshape(previous_memory, shape)
        previous_output = tf.reshape(previous_output, shape)

      with tf.variable_scope('Convolve'):
        channels = input.get_shape()[-1].value
        filters = self._filters
        gates = 4 * filters if filters > 1 else 4
        x = tf.concat([input, previous_output], axis=3)
        n = channels + filters
        m = gates
        W = tf.get_variable('Weights', self._kernel + [n, m], initializer=self._initializer)
        y = tf.nn.convolution(x, W, 'SAME')
        y += tf.get_variable('Biases', [m], initializer=tf.constant_initializer(0.0))
        input, input_gate, forget_gate, output_gate = tf.split(y, 4, axis=3)

      with tf.variable_scope('LSTM'):
        memory = (previous_memory
          * tf.sigmoid(forget_gate + self._forget_bias)
          + tf.sigmoid(input_gate) * self._activation(input))
        output = self._activation(memory) * tf.sigmoid(output_gate)

      with tf.variable_scope('Flatten'):
        shape = [-1, self._size]
        output = tf.reshape(output, shape)
        memory = tf.reshape(memory, shape)

      return output, tf.contrib.rnn.LSTMStateTuple(memory, output)


def flatten(tensor):
  samples, timesteps, height, width, filters = tensor.get_shape().as_list()
  return tf.reshape(tensor, [samples, timesteps, height * width * filters])


def expand(tensor, height, width, filters):
  samples, timesteps, features = tensor.get_shape().as_list()
  return tf.reshape(tensor, [samples, timesteps, height, width, filters])
