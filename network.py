import tensorflow as tf
import numpy as np


class Network(object):
    def __init__(self):
        self.name = "Unnamed"
        self.layers = []

        self.input_shape = None # w x h x channels
        self.output_shape = None # [2] one hot vector for classes

    def build_net(self, input):
        raise Exception("Need to be implemented in subclass!")

    def create_variable(self, name, shape):
        return tf.get_variable(name, shape, initializer=tf.random_normal_initializer())

    def add_layer(self, name, layer):
        self.layers.append((name, layer))

    def get_unique_name(self, prefix):
        id = sum(t.startswith(prefix) for t, _ in self.layers) + 1
        return '%s_%d' % (prefix, id)

    def get_last_output(self):
        return self.layers[-1][1]

    def input(self, input):
        self.add_layer("input", input)
        return self

    def conv(self, kx, ky, sx, sy, in_size, out_size, name=None):
        name = name or self.get_unique_name("conv")

        with tf.variable_scope(name) as scope:
            input = self.get_last_output()
            kernel = self.create_variable("weights", [kx, ky, in_size, out_size])
            bias = self.create_variable("bias", [out_size])

            conv = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(input, kernel, strides=[1, sx, sy, 1], padding='SAME'), bias),
                              name=scope.name)
            self.add_layer(name, conv)

        return self

    def pool(self, kx, ky, sx=1, sy=1, name=None):
        name = name or self.get_unique_name("pool")

        input = self.get_last_output()
        pool = tf.nn.max_pool(input, ksize=[1, kx, ky, 1], strides=[1, sx, sy, 1], padding='SAME')
        self.add_layer(name, pool)

        return self

    def fc(self, out_size, name=None):
        name = name or self.get_unique_name("fc")

        with tf.variable_scope(name) as scope:
            input = self.get_last_output()

            shape = input.get_shape().as_list()
            in_size = np.prod(shape[1:])

            weights = self.create_variable("weights", [in_size, out_size])
            bias = self.create_variable("bias", [out_size])

            input_flat = tf.reshape(input, [-1, in_size])
            fc = tf.nn.relu(tf.nn.xw_plus_b(input_flat, weights, bias, name=scope.name))

            self.add_layer(name, fc)

        return self

    def lrn(self, depth_radius = 5, bias=1.0, alpha=0.0005, beta=0.75, name=None):
        name = name or self.get_unique_name("lrn")

        input = self.get_last_output()
        lrn = tf.nn.lrn(input, depth_radius, bias=bias, alpha=alpha, beta=beta)
        self.add_layer(name, lrn)

        return self

    def dropout(self, dropout_rate, name=None):
        name = name or self.get_unique_name("dropout")

        input = self.get_last_output()
        dropout = tf.nn.dropout(input, dropout_rate)
        self.add_layer(name, dropout)

        return self

    def output(self, out_size, name=None):

        name = name or self.get_unique_name("output")

        with tf.variable_scope(name) as scope:
            input = self.get_last_output()

            shape = input.get_shape().as_list()
            in_size = np.prod(shape[1:])

            weights = self.create_variable("weights", [in_size, out_size])
            bias = self.create_variable("bias", [out_size])

            input_flat = tf.reshape(input, [-1, in_size])
            fc = tf.nn.xw_plus_b(input_flat, weights, bias, name=scope.name)

            self.add_layer(name, fc)

        return self

    def softmax(self, name=None):
        name = name or self.get_unique_name("output")

        input = self.get_last_output()
        softmax = tf.nn.softmax(input)
        self.add_layer(name, softmax)

        return self

    def debug(self):

        for name, layer in self.layers:
            print name, layer.get_shape()

        return self
