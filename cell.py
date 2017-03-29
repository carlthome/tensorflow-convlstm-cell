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

  def __call__(self, x, h, scope=None):
    with tf.variable_scope(scope or self.__class__.__name__):
      previous_memory, previous_output = h

      with tf.variable_scope('Expand'):
        samples = x.get_shape()[0].value
        shape = [samples, self._height, self._width, -1]
        x = tf.reshape(x, shape)
        previous_memory = tf.reshape(previous_memory, shape)
        previous_output = tf.reshape(previous_output, shape)

      with tf.variable_scope('Convolve'):
        channels = x.get_shape()[-1].value
        filters = self._filters
        gates = 4 * filters if filters > 1 else 4
        x = tf.concat([x, previous_output], axis=3)
        n = channels + filters
        m = gates
        W = tf.get_variable('kernel', self._kernel + [n, m], initializer=self._initializer)
        y = tf.nn.convolution(x, W, 'SAME')
        y += tf.get_variable('bias', [m], initializer=tf.constant_initializer(0.0))
        input_contribution, input_gate, forget_gate, output_gate = tf.split(y, 4, axis=3)

      with tf.variable_scope('LSTM'):
        memory = (previous_memory
          * tf.sigmoid(forget_gate + self._forget_bias)
          + tf.sigmoid(input_gate) * self._activation(input_contribution))
        output = self._activation(memory) * tf.sigmoid(output_gate)

      with tf.variable_scope('Flatten'):
        shape = [-1, self._size]
        output = tf.reshape(output, shape)
        memory = tf.reshape(memory, shape)

      return output, tf.contrib.rnn.LSTMStateTuple(memory, output)


class ConvGRUCell(tf.contrib.rnn.RNNCell):
  """A GRU cell with convolutions instead of multiplications."""

  def __init__(self, height, width, filters, kernel, initializer=None, activation=tf.tanh):
    self._height = height
    self._width = width
    self._filters = filters
    self._kernel = kernel
    self._initializer = initializer
    self._activation = activation
    self._size = int(self._height * self._width * self._filters)

  @property
  def state_size(self):
    return self._size

  @property
  def output_size(self):
    return self._size

  def __call__(self, x, h, scope=None):
    with tf.variable_scope(scope or self.__class__.__name__):

      with tf.variable_scope('Expand'):
        samples = x.get_shape()[0].value
        shape = [samples, self._height, self._width, -1]
        x = tf.reshape(x, shape)
        h = tf.reshape(h, shape)

      with tf.variable_scope('Gates'):
        channels = x.get_shape()[-1].value
        inputs = tf.concat([x, h], axis=3)
        n = channels + self._filters
        m = 2 * self._filters if self._filters > 1 else 2
        W = tf.get_variable('kernel', self._kernel + [n, m], initializer=self._initializer)
        y = tf.nn.convolution(inputs, W, 'SAME')
        y += tf.get_variable('bias', [m], initializer=tf.constant_initializer(1.0))
        reset_gate, update_gate = tf.split(y, 2, axis=3)
        reset_gate, update_gate = tf.sigmoid(reset_gate), tf.sigmoid(update_gate)

      with tf.variable_scope('Output'):
        inputs = tf.concat([x, reset_gate * h], axis=3)
        n = channels + self._filters
        m = self._filters
        W = tf.get_variable('kernel', self._kernel + [n, m], initializer=self._initializer)
        y = tf.nn.convolution(inputs, W, 'SAME')
        y += tf.get_variable('bias', [m], initializer=tf.constant_initializer(0.0))
        y = self._activation(y)
        output = update_gate * h + (1 - update_gate) * y

      with tf.variable_scope('Flatten'):
        output = tf.reshape(output, [-1, self._size])

      return output, output


def flatten(tensor):
  samples, timesteps, height, width, filters = tensor.get_shape().as_list()
  return tf.reshape(tensor, [samples, timesteps, height * width * filters])


def expand(tensor, height, width, filters):
  samples, timesteps, features = tensor.get_shape().as_list()
  return tf.reshape(tensor, [samples, timesteps, height, width, filters])
