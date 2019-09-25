import tensorflow as tf

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.python.ops.math_ops import tanh
from tensorflow.python.ops import init_ops

from tensorflow.python.util import nest
import collections


class ConvLSTMCell(object):
    """ Convolutional LSTM network cell (ConvLSTMCell).
    The implementation is based on http://arxiv.org/abs/1506.04214.
     and `BasicLSTMCell` in TensorFlow.
     2D version: https://github.com/iwyoo/ConvLSTMCell-tensorflow
    """

    def __init__(self, hidden_num, filter_size=None,
                 forget_bias=1.0, activation=tanh, name="ConvLSTMCell"):
        if filter_size is None:
            filter_size = [3, 3, 3]
        self.hidden_num = hidden_num
        self.filter_size = filter_size
        self.forget_bias = forget_bias
        self.activation = activation
        self.name = name

    def zero_state(self, batch_size, height, width, depth):
        # return tf.zeros_like(tf.placeholder(tf.float32, shape=[batch_size, height, width, depth, self.hidden_num * 2]))



        # A = tf.tile(tf.expand_dims(input_tensor, axis=5), [1, 1, 1, 1, 1, self.hidden_num * 2])
        # return tf.zeros_like(tf.placeholder(tf.float32, shape=[batch_size, height, width, depth, self.hidden_num * 2]))

        # return tf.keras.backend.zeros(shape=[batch_size, height, width, depth, self.hidden_num * 2])
        return tf.zeros([batch_size, height, width, depth, self.hidden_num * 2])

    def __call__(self, inputs, state, scope=None):
        """Convolutional Long short-term memory cell (ConvLSTM)."""
        with vs.variable_scope(scope or self.name):  # "ConvLSTMCell"
            # c, h = array_ops.split(3, 2, state) original one
            c, h = array_ops.split(state, 2, axis=4)  # Hessam

            # batch_size * height * width * channel
            concat = _conv([inputs, h], 4 * self.hidden_num, self.filter_size)

            # i = input_gate, j = new_input, f = forget_gate, o = output_gate
            i, j, f, o = array_ops.split(concat, 4, axis=4)

            new_c = (c * sigmoid(f + self.forget_bias) + sigmoid(i) *
                     self.activation(j))
            new_h = self.activation(new_c) * sigmoid(o)
            new_state = array_ops.concat([new_c, new_h], axis=4)

            return new_h, new_state


def _conv(args, output_size, filter_size, stddev=0.001, bias=True, bias_start=0.0, scope=None):
    if args is None or (nest.is_sequence(args) and not args):
        raise ValueError("`args` must be specified")
    if not nest.is_sequence(args):
        args = [args]

    # Calculate the total size of arguments on dimension 3.
    # (batch_size x height x width x arg_size)
    total_arg_size = 0
    shapes = [a.get_shape().as_list() for a in args]
    height = shapes[0][1]
    width = shapes[0][2]
    depth = shapes[0][3]
    for shape in shapes:
        if len(shape) != 5:
            raise ValueError("Conv is expecting 4D arguments: %s" % str(shapes))
        if not shape[4]:
            raise ValueError("Conv expects shape[4] of arguments: %s" % str(shapes))
        if shape[1] == height and shape[2] == width and shape[3] == depth:
            total_arg_size += shape[4]
        else:
            raise ValueError("Inconsistent height and width size in arguments: %s" % str(shapes))

    with vs.variable_scope(scope or "Conv"):
        kernel = vs.get_variable("Kernel",
                                 [filter_size[0], filter_size[1], filter_size[2],  total_arg_size, output_size],
                                 initializer=init_ops.truncated_normal_initializer(stddev=stddev), trainable=True)

        if len(args) == 1:
            res = tf.nn.conv3d(args[0], kernel, [1, 1, 1, 1, 1], padding='SAME')
        else:
            res = tf.nn.conv3d(array_ops.concat(args, 4), kernel, [1, 1, 1, 1, 1], padding='SAME')

        if not bias: return res
        bias_term = vs.get_variable("Bias", [output_size],
                                    initializer=init_ops.constant_initializer(bias_start))
    return res + bias_term
