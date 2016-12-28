# TensorFlow ConvLSTM Cell [![Build Status](https://travis-ci.org/carlthome/tensorflow-convlstm-cell.svg?branch=master)](https://travis-ci.org/carlthome/tensorflow-convlstm-cell)
A ConvLSTM cell for TensorFlow's RNN API. 

# About TensorFlow's RNNs
`tf.nn.dynamic_rnn` requires the input/output shape to be unchanged over time, which becomes problematic for a ConvLSTM because the first iteration will increase the number of channels to the number of filters. In order to get around this, one could do an initial 3D convolution to match the number of filters before calling the RNN API.

`tf.nn.dynamic_rnn` also requires input to be 3D tensors `(sequence, time, feature)`, while a ConvLSTM takes 5D tensors `(sequence, time, width, height, channel)`. A way of getting around this is to flatten the input and expand the output with reshaping. 

Therefore this implementation provides three utility functions (`conv_3d`, `flatten` and `expand`) to deal with this.

# Usage
```py
import tensorflow as tf

from ConvLSTMCell import ConvLSTMCell, conv_3d, flatten, expand

batch_size = 32
timesteps = 100
width = 16
height = 16
channels = 3
filters = 12

# Create a placeholder for videos.
inputs = tf.placeholder(tf.float32, [batch_size, timesteps, width, height, channels])

# 3D convolve videos to match the number of filters in the ConvLSTM.
inputs = conv_3d(inputs, filters)

# Flatten input because tf.nn.dynamic_rnn only accepts 3D input.
inputs = flatten(inputs)

# Add the ConvLSTM step.
cell = ConvLSTMCell(height, width, filters)
cell = tf.nn.rnn_cell.MultiRNNCell([cell] * 3)
cell = tf.nn.rnn_cell.DropoutWrapper(cell, 0.5, 0.5)
state = cell.zero_state(batch_size, inputs.dtype)
outputs, state = tf.nn.dynamic_rnn(cell, inputs, initial_state=state)

# Reshape outputs to videos again, because tf.nn.dynamic_rnn only accepts 3D input.
outputs = expand(outputs, height, width)
```
