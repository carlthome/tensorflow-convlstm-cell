import tensorflow as tf

class ConvLSTMCell(tf.contrib.rnn.RNNCell):
  """A LSTM cell with convolutions instead of multiplications.

  Reference:
    Xingjian, S. H. I., et al. "Convolutional LSTM network: A machine learning approach for precipitation nowcasting." Advances in Neural Information Processing Systems. 2015.
  """

  def __init__(self, shape, filters, kernel, initializer=None, forget_bias=1.0, activation=tf.tanh, normalize=True):
    self._kernel = kernel
    self._filters = filters
    self._initializer = initializer
    self._forget_bias = forget_bias
    self._activation = activation
    self._size = tf.TensorShape(shape + [self._filters])
    self._normalize = normalize
    self._feature_axis = self._size.ndims

  @property
  def state_size(self):
    return tf.contrib.rnn.LSTMStateTuple(self._size, self._size)

  @property
  def output_size(self):
    return self._size

  def __call__(self, x, h, scope=None):
    with tf.variable_scope(scope or self.__class__.__name__):
      previous_memory, previous_output = h

      channels = x.shape[-1].value
      filters = self._filters
      gates = 4 * filters if filters > 1 else 4
      x = tf.concat([x, previous_output], axis=self._feature_axis)
      n = channels + filters
      m = gates
      W = tf.get_variable('kernel', self._kernel + [n, m], initializer=self._initializer)
      y = tf.nn.convolution(x, W, 'SAME')
      if not self._normalize:
        y += tf.get_variable('bias', [m], initializer=tf.constant_initializer(0.0))
      input_contribution, input_gate, forget_gate, output_gate = tf.split(y, 4, axis=self._feature_axis)

      if self._normalize:
        input_contribution = tf.contrib.layers.layer_norm(input_contribution)
        input_gate = tf.contrib.layers.layer_norm(input_gate)
        forget_gate = tf.contrib.layers.layer_norm(forget_gate)
        output_gate = tf.contrib.layers.layer_norm(output_gate)

      memory = (previous_memory
        * tf.sigmoid(forget_gate + self._forget_bias)
        + tf.sigmoid(input_gate) * self._activation(input_contribution))

      if self._normalize:
        memory = tf.contrib.layers.layer_norm(memory)

      output = self._activation(memory) * tf.sigmoid(output_gate)

      return output, tf.contrib.rnn.LSTMStateTuple(memory, output)


class ConvGRUCell(tf.contrib.rnn.RNNCell):
  """A GRU cell with convolutions instead of multiplications."""

  def __init__(self, shape, filters, kernel, initializer=None, activation=tf.tanh, normalize=True):
    self._filters = filters
    self._kernel = kernel
    self._initializer = initializer
    self._activation = activation
    self._size = tf.TensorShape(shape + [self._filters])
    self._normalize = normalize
    self._feature_axis = self._size.ndims

  @property
  def state_size(self):
    return self._size

  @property
  def output_size(self):
    return self._size

  def __call__(self, x, h, scope=None):
    with tf.variable_scope(scope or self.__class__.__name__):

      with tf.variable_scope('Gates'):
        channels = x.shape[-1].value
        inputs = tf.concat([x, h], axis=self._feature_axis)
        n = channels + self._filters
        m = 2 * self._filters if self._filters > 1 else 2
        W = tf.get_variable('kernel', self._kernel + [n, m], initializer=self._initializer)
        y = tf.nn.convolution(inputs, W, 'SAME')
        if self._normalize:
          reset_gate, update_gate = tf.split(y, 2, axis=self._feature_axis)
          reset_gate = tf.contrib.layers.layer_norm(reset_gate)
          update_gate = tf.contrib.layers.layer_norm(update_gate)
        else:
          y += tf.get_variable('bias', [m], initializer=tf.constant_initializer(1.0))
          reset_gate, update_gate = tf.split(y, 2, axis=self._feature_axis)
        reset_gate, update_gate = tf.sigmoid(reset_gate), tf.sigmoid(update_gate)

      with tf.variable_scope('Output'):
        inputs = tf.concat([x, reset_gate * h], axis=self._feature_axis)
        n = channels + self._filters
        m = self._filters
        W = tf.get_variable('kernel', self._kernel + [n, m], initializer=self._initializer)
        y = tf.nn.convolution(inputs, W, 'SAME')
        if self._normalize:
          y = tf.contrib.layers.layer_norm(y)
        else:
          y += tf.get_variable('bias', [m], initializer=tf.constant_initializer(0.0))
        y = self._activation(y)
        output = update_gate * h + (1 - update_gate) * y

      return output, output
