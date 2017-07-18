#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Patrick Wieschollek <mail@patwie.com>

from abc import ABCMeta, abstractmethod, abstractproperty
import tensorflow as tf
from tensorpack import *
from tensorpack.tfutils.scope_utils import auto_reuse_variable_scope

"""
I highly suggest to read

http://colah.github.io/posts/2015-08-Understanding-LSTMs/
"""


class ConvRNNCell(object):
    __metaclass__ = ABCMeta

    def __init__(self, tensor_shape, out_channel, kernel_shape, nl=tf.nn.tanh, normalize_fn=None):
        """Abstract representation for 2D recurrent cells.

        Args:
            tensor_shape: shape of inputs (must be fully specified)
            out_channel: number of output channels
            kernel_shape: size of filters
            nl (TYPE, optional): non-linearity (default: tf.nn.tanh)
            normalize_fn (None, optional): normalization steps (e.g. tf.contrib.layers.layer_norm)
        """
        super(ConvRNNCell, self).__init__()
        self.state_tensor = None

        assert len(tensor_shape), "tensor_shape should have 4 dims [BHWC]"

        self.input_shape = tensor_shape
        self.out_channel = out_channel
        self.kernel_shape = kernel_shape

        self.nl = nl
        self.normalize_fn = normalize_fn

    @abstractproperty
    def default_state(self):
        pass

    def state(self):
        if self.state_tensor is None:
            self.state_tensor = self.default_state()
        return self.state_tensor

    @abstractmethod
    def _calc(self, tensor):
        pass

    def __call__(self, tensor):
        return self._calc(tensor)


class ConvLSTMCell(ConvRNNCell):
    """Represent LSTM-layer using convolutions.

    conv_gates:
        i = sigma(x*U1 + s*W1)  input gate
        f = sigma(x*U2 + s*W2)  forget gate
        o = sigma(x*U3 + s*W3)  output gate
        g = tanh(x*U4 + s*W4)   candidate hidden state

    memory update:
        c = c * f + g * i       internal memory
    s = tanh(c) * o         output hiden state

    """
    def default_state(self):
        b, h, w, c = self.input_shape
        return (tf.zeros([b, h, w, self.out_channel]), tf.zeros([b, h, w, self.out_channel]))

    @auto_reuse_variable_scope
    def _calc(self, x):
        c, s = self.state()

        xs = tf.concat(axis=3, values=[x, s])
        igfo = Conv2D('conv_gates', xs, 4 * self.out_channel, self.kernel_shape,
                      nl=tf.identity, use_bias=(self.normalize_fn is None))
        # i = input_gate, g = hidden state, f = forget_gate, o = output_gate
        i, g, f, o = tf.split(axis=3, num_or_size_splits=4, value=igfo)

        if self.normalize_fn is not None:
            i, g = self.normalize_fn(i), self.normalize_fn(g)
            f, o = self.normalize_fn(f), self.normalize_fn(o)

        i, g = tf.nn.sigmoid(i), self.nl(g)
        f, o = tf.nn.sigmoid(f), tf.nn.sigmoid(o)

        # memory update
        c = c * f + g * i
        if self.normalize_fn is not None:
            c = self.normalize_fn(c)

        # output
        s = self.nl(c) * tf.nn.sigmoid(o)
        self.state_tensor = (c, s)

        return s


class ConvGRUCell(ConvRNNCell):
    """Represent GRU-layer using convolutions.

    z = sigma(x*U1 + s*W1)     update gate
    r = sigma(x*U2 + s*W2)     reset gate
    h = tanh(x*U3 + (s*r)*W3)
    s = (1-z)*h + z*s
    """
    def default_state(self):
        """GRU just uses the output as the state for the next computation.
        """
        b, h, w, c = self.input_shape
        return tf.zeros([b, h, w, self.out_channel])

    @auto_reuse_variable_scope
    def _calc(self, x):
        s = self.state()

        # we concat x and s to reduce the number of conv-calls
        xs = tf.concat(axis=3, values=[x, s])
        zr = Conv2D('conv_zr', xs, 2 * self.out_channel, self.kernel_shape,
                    nl=tf.identity, use_bias=(self.normalize_fn is None))

        # z (update gate), r (reset gate)
        z, r = tf.split(axis=3, num_or_size_splits=2, value=zr)

        if self.normalize_fn is not None:
            r, z = self.normalize_fn(r), self.normalize_fn(z)

        r, z = tf.sigmoid(r), tf.sigmoid(z)

        h = tf.concat(axis=3, values=[x, s * r])
        h = Conv2D('conv_h', h, self.out_channel, self.kernel_shape,
                   nl=tf.identity, use_bias=(self.normalize_fn is None))

        if self.normalize_fn is not None:
            h = self.normalize_fn(h)

        h = self.nl(h)
        s = (1 - z) * h + z * s

        self.state_tensor = s

        return s


@layer_register()
def ConvRNN(x, cell):
    assert len(x.get_shape().as_list()) == 4, "input in ConvRNN should be B,H,W,C"
    return cell(x)


@layer_register()
def ConvRNN_unroll(x, cell):
    assert len(x.get_shape().as_list()) == 5, "input in ConvRNN should be B,T,H,W,C"
    time_dim = x.get_shape().as_list()[1]

    outputs = []
    for t in range(time_dim):
        outputs.append(cell(x[:, t, :, :, :]))

    return tf.stack(outputs, axis=1)
