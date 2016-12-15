# tensorflow-convlstm-cell
A ConvLSTM cell for TensorFlow's RNN API.

# Usage
```py
import tensorflow as tf
from ConvLSTMCell import ConvLSTMCell, convolve_inputs, expand_outputs

batch_size = 32
timesteps = 100
width = 16
height = 16
channels = 3
filters = 12

inputs = tf.placeholder(tf.float32, [batch_size, timesteps, width, height, channels])

inputs = convolve_inputs(inputs, batch_size, height, width, channels, filters)
  
cell = ConvLSTMCell(filters, height, width, channels)
cell = tf.nn.rnn_cell.MultiRNNCell([cell] * 3)
cell = tf.nn.rnn_cell.DropoutWrapper(cell, 0.5, 0.5)
state = cell.zero_state(batch_size, tf.float32)
outputs, state = tf.nn.dynamic_rnn(cell=cell, inputs=inputs, initial_state=state)

outputs = expand_outputs(outputs, batch_size, height, width, filters)
```
