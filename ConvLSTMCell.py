from tensorflow.python.ops.variable_scope import variable_scope, get_variable
from tensorflow.python.ops.init_ops import constant_initializer
from tensorflow.python.ops.math_ops import tanh, sigmoid
from tensorflow.python.ops.rnn_cell import RNNCell, LSTMStateTuple
from tensorflow.python.ops.nn_ops import conv2d, conv3d, bias_add
from tensorflow.python.ops.array_ops import concat, split, reshape, zeros


class ConvLSTMCell(RNNCell):
  """A LSTM cell with convolutions instead of multiplications.

  Reference:
    Xingjian, S. H. I., et al. "Convolutional LSTM network: A machine learning approach for precipitation nowcasting." Advances in Neural Information Processing Systems. 2015.
  """
  def __init__(self, filters, height, width, channels, kernel=[3, 3], forget_bias=1.0, activation=tanh):
    self._kernel = kernel
    self._num_units = filters
    self._height = height
    self._width = width
    self._channels = channels
    self._forget_bias = forget_bias
    self._activation = activation

  @property
  def state_size(self):
    size = self._height * self._width * self._num_units
    return LSTMStateTuple(size, size)

  @property
  def output_size(self):
    return self._height * self._width * self._num_units
    
  def zero_state(self, batch_size, dtype):
    shape = [batch_size, self._height * self._width * self._num_units]
    memory = zeros(shape, dtype=dtype)
    output = zeros(shape, dtype=dtype)
    return LSTMStateTuple(memory, output)

  def __call__(self, input, state, scope=None):    
    with variable_scope(scope or 'ConvLSTMCell'):
      previous_memory, previous_output = state
 
      with variable_scope('Expand'):
        shape =  [-1, self._height, self._width, self._num_units]
        input = reshape(input, shape)
        previous_memory = reshape(previous_memory, shape)
        previous_output = reshape(previous_output, shape)

      with variable_scope('Convolve'):
        x = concat(3, [input, previous_output])
        W = get_variable('Weights', self._kernel + [2 * self._num_units, 4 * self._num_units])
        b = get_variable('Biases', [4 * self._num_units], initializer=constant_initializer(0.0))
        y = bias_add(conv2d(x, W, [1] * 4, 'SAME'), b)
        input_gate, new_input, forget_gate, output_gate = split(3, 4, y)

      with variable_scope('LSTM'):
        memory = (previous_memory
          * sigmoid(forget_gate + self._forget_bias)
          + sigmoid(input_gate) * self._activation(new_input))
        output = self._activation(memory) * sigmoid(output_gate)
   
      with variable_scope('Flatten'):
        shape = [-1, self._height * self._width * self._num_units]
        output = reshape(output, shape)
        memory = reshape(memory, shape)

      return output, LSTMStateTuple(memory, output)
   

def convolve_inputs(inputs, filters, kernel=[1, 1, 1], scope=None):
  s = inputs.get_shape()
  samples = s[0].value
  timesteps = s[1].value
  height = s[2].value
  width = s[3].value
  channels = s[4].value
  
  with variable_scope('Conv3D'):
    W = get_variable('Weights', kernel + [channels, filters])
    b = get_variable('Biases', [filters], initializer=constant_initializer(0.0))
    y = bias_add(conv3d(inputs, W, [1] * 5, 'SAME'), b)
    return reshape(y, [samples, timesteps, height * width * filters])
    

def expand_outputs(outputs, height, width):
  s = outputs.get_shape()
  samples = s[0].value
  timesteps = s[1].value
  
  return reshape(outputs, [samples, timesteps, height, width, -1])
