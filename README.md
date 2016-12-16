# TensorFlow ConvLSTM Cell [![Build Status](https://travis-ci.org/carlthome/tensorflow-convlstm-cell.svg?branch=master)](https://travis-ci.org/carlthome/tensorflow-convlstm-cell)
A ConvLSTM cell for TensorFlow's RNN API. 

# About TensorFlow's RNNs
`tf.nn.dynamic_rnn` requires input to be 3D tensors `(sequence, time, feature)`, while a ConvLSTM takes 5D tensors `(sequence, time, width, height, channel)`. A way of getting around this is to flatten the input and expand the output with reshaping. 

`tf.nn.dynamic_rnn` also requires the input/output shape to be unchanged over time, which becomes problematic for a ConvLSTM because the first iteration will increase the number of channels to the number of filters. In order to get around this, one could do an initial 3D convolution to match the number of filters before calling the RNN API.

Therefore this implementation provides two utility functions, `convolve_inputs` and `expand_outputs`, to deal with this.

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

# Create a placeholder for video sequences.
inputs = tf.placeholder(tf.float32, [batch_size, timesteps, width, height, channels])

# 3D convolve video sequences to match the number of filters in the ConvLSTM.
inputs = convolve_inputs(inputs, filters)

# Add the ConvLSTM step.
cell = ConvLSTMCell(filters, height, width, channels)
cell = tf.nn.rnn_cell.MultiRNNCell([cell] * 3)
cell = tf.nn.rnn_cell.DropoutWrapper(cell, 0.5, 0.5)
state = cell.zero_state(batch_size, tf.float32)
outputs, state = tf.nn.dynamic_rnn(cell=cell, inputs=inputs, initial_state=state)

# Reshape outputs to videos again, because tf.nn.dynamic_rnn only accepts 3D input.
outputs = expand_outputs(outputs, height, width)
```
