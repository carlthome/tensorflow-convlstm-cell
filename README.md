# TensorFlow ConvLSTM Cell [![Build Status](https://travis-ci.org/carlthome/tensorflow-convlstm-cell.svg?branch=master)](https://travis-ci.org/carlthome/tensorflow-convlstm-cell)
A ConvLSTM cell for TensorFlow's RNN API.

# About TensorFlow's RNNs
`tf.nn.dynamic_rnn` requires input to be 3D tensors `(sequence, time, feature)`, while a ConvLSTM takes 5D tensors `(sequence, time, width, height, channel)`. A way of getting around this is to flatten the input and expand the output with reshaping. Therefore this implementation provides two utility functions (`flatten` and `expand`) to deal with this.

# Usage
```python
import tensorflow as tf

from cell import ConvLSTMCell, flatten, expand

batch_size = 32
timesteps = 100
width = 16
height = 16
channels = 3
filters = 12
kernel = [3, 3]

# Create a placeholder for videos.
inputs = tf.placeholder(tf.float32, [batch_size, timesteps, width, height, channels])

# Flatten input because tf.nn.dynamic_rnn only accepts 3D input.
inputs = flatten(inputs)

# Add the ConvLSTM step.
cell = ConvLSTMCell(height, width, filters, kernel)
outputs, state = tf.nn.dynamic_rnn(cell, inputs, dtype=inputs.dtype)

# Reshape outputs to videos again, because tf.nn.dynamic_rnn only accepts 3D input.
outputs = expand(outputs, height, width, filters)

# There's also a ConvGRUCell that is more memory efficient.
from cell import ConvGRUCell
cell = ConvGRUCell(height, width, filters, kernel)
outputs, state = tf.nn.dynamic_rnn(cell, inputs, dtype=inputs.dtype)
```
