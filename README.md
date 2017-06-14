# TensorFlow ConvLSTM Cell [![Build Status](https://travis-ci.org/carlthome/tensorflow-convlstm-cell.svg?branch=master)](https://travis-ci.org/carlthome/tensorflow-convlstm-cell)
A ConvLSTM cell for TensorFlow's RNN API.

```python
import tensorflow as tf

batch_size = 32
timesteps = 100
shape = [640, 480]
kernel = [3, 3]
channels = 3
filters = 12

# Create a placeholder for videos.
inputs = tf.placeholder(tf.float32, [batch_size, timesteps] + shape + [channels])

# Add the ConvLSTM step.
from cell import ConvLSTMCell
cell = ConvLSTMCell(shape, filters, kernel)
outputs, state = tf.nn.dynamic_rnn(cell, inputs, dtype=inputs.dtype, time_major=True)

# There's also a ConvGRUCell that is more memory efficient.
from cell import ConvGRUCell
cell = ConvGRUCell(shape, filters, kernel)
outputs, state = tf.nn.dynamic_rnn(cell, inputs, dtype=inputs.dtype, time_major=True)

# It's also possible to enter 2D input or 4D input instead of 3D.
features = 100
inputs = tf.placeholder(tf.float32, [timesteps, batch_size, features, channels])
cell = ConvLSTMCell([features], filters, kernel=[3])
outputs, state = tf.nn.bidirectional_dynamic_rnn(cell, cell, inputs, dtype=inputs.dtype, time_major=True)

shape = [50, 50, 50]
kernel = [1, 3, 5]
inputs = tf.placeholder(tf.float32, [timesteps, batch_size] + shape + [channels])
cell = ConvGRUCell(shape, filters, kernel)
outputs, state= tf.nn.bidirectional_dynamic_rnn(cell, cell, inputs, dtype=inputs.dtype, time_major=True)
```
