# TensorFlow ConvLSTM Cell [![Build Status](https://travis-ci.org/carlthome/tensorflow-convlstm-cell.svg?branch=master)](https://travis-ci.org/carlthome/tensorflow-convlstm-cell)
A ConvLSTM cell for TensorFlow's RNN API.

# Usage
```python
import tensorflow as tf
from cell import ConvLSTMCell

batch_size = 32
timesteps = 100
shape = [640, 480]
kernel = [3, 3]
channels = 3
filters = 12

# Create a placeholder for videos.
inputs = tf.placeholder(tf.float32, [batch_size, timesteps] + shape + [channels])

# Make placeholder time major for RNN. (see https://github.com/tensorflow/tensorflow/pull/5142)
inputs = tf.transpose(inputs, (1, 0, 2, 3, 4))

# Add the ConvLSTM step.
cell = ConvLSTMCell(shape, filters, kernel)
outputs, state = tf.nn.dynamic_rnn(cell, inputs, dtype=inputs.dtype, time_major=True)

# Make placeholder batch major again after RNN. (see https://github.com/tensorflow/tensorflow/pull/5142)
outputs = tf.transpose(inputs, (1, 0, 2, 3, 4))

# There's also a ConvGRUCell that is more memory efficient.
from cell import ConvGRUCell
cell = ConvGRUCell(shape, filters, kernel)
outputs, state = tf.nn.dynamic_rnn(cell, inputs, dtype=inputs.dtype, time_major=True)
```
