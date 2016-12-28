from tensorflow.python.ops.array_ops import concat, reshape, split, zeros
from tensorflow.python.ops.init_ops import zeros_initializer, orthogonal_initializer
from tensorflow.python.ops.math_ops import sigmoid, tanh
from tensorflow.python.ops.nn_ops import bias_add, conv2d, conv3d
from tensorflow.python.ops.rnn_cell import LSTMStateTuple, RNNCell
from tensorflow.python.ops.variable_scope import get_variable, variable_scope


class ConvLSTMCell(RNNCell):
  """A LSTM cell with convolutions instead of multiplications.

  Reference: 
    Xingjian, S. H. I., et al. "Convolutional LSTM network: A machine learning approach for precipitation nowcasting." Advances in Neural Information Processing Systems. 2015.
  """

  def __init__(self, height, width, filters, kernel=[3, 3], forget_bias=1.0, activation=tanh, weights_initializer=orthogonal_initializer()):
    self._height = height
    self._width = width
    self._filters = filters
    self._kernel = kernel
    self._forget_bias = forget_bias
    self._activation = activation
    self._weights_initializer = weights_initializer

  @property
  def state_size(self):
    size = self._height * self._width * self._filters
    return LSTMStateTuple(size, size)

  @property
  def output_size(self):
    return self._height * self._width * self._filters

  def __call__(self, input, state, scope=None):
    with variable_scope(scope or 'ConvLSTMCell'):
      previous_memory, previous_output = state

      with variable_scope('Expand'):
        shape = [-1, self._height, self._width, self._filters]
        input = reshape(input, shape)
        previous_memory = reshape(previous_memory, shape)
        previous_output = reshape(previous_output, shape)

      with variable_scope('Convolve'):
        x = concat(3, [input, previous_output])
        W = get_variable('Weights', self._kernel + [2 * self._filters, 4 * self._filters], initializer=self._weights_initializer)
        b = get_variable('Biases', [4 * self._filters], initializer=zeros_initializer)
        y = bias_add(conv2d(x, W, [1] * 4, 'SAME'), b)
        input_gate, new_input, forget_gate, output_gate = split(3, 4, y)

      with variable_scope('LSTM'):
        memory = (previous_memory 
          * sigmoid(forget_gate + self._forget_bias) 
          + sigmoid(input_gate) * self._activation(new_input))
        output = self._activation(memory) * sigmoid(output_gate)

      with variable_scope('Flatten'):
        shape = [-1, self._height * self._width * self._filters]
        output = reshape(output, shape)
        memory = reshape(memory, shape)

      return output, LSTMStateTuple(memory, output)


def conv_3d(tensor, filters, kernel=[1, 1, 1], weights_initializer=orthogonal_initializer(), scope=None):
  samples, timesteps, height, width, channels = tensor.get_shape().as_list()
  with variable_scope(scope or 'Conv3D'):
    W = get_variable('Weights', kernel + [channels, filters], initializer=weights_initializer)
    b = get_variable('Biases', [filters], initializer=zeros_initializer)
    y = bias_add(conv3d(tensor, W, [1] * 5, 'SAME'), b)
    return y


def flatten(tensor):
  samples, timesteps, height, width, filters = tensor.get_shape().as_list()
  return reshape(tensor, [samples, timesteps, height * width * filters])


def expand(tensor, height, width):
  samples, timesteps, features = tensor.get_shape().as_list()
  return reshape(tensor, [samples, timesteps, height, width, -1])
