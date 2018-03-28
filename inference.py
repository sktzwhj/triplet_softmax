import tensorflow as tf
import numpy as np


class siamese:
    # Create model
    def __init__(self):
        self.x1 = tf.placeholder(tf.float32, [None, 10])
        self.x2 = tf.placeholder(tf.float32, [None, 10])
        self.x3 = tf.placeholder(tf.float32, [None, 10])
        self.triplet_num = tf.placeholder(tf.float32, [None, 1])

        with tf.variable_scope("siamese") as scope:
            self.o1 = self.network(self.x1)
            scope.reuse_variables()
            self.o2 = self.network(self.x2)
            scope.reuse_variables()
            self.o3 = self.network(self.x3)

        self.y_ = tf.placeholder(tf.float32, [None])
        self.loss = self.triplet_loss()

    def entropy_shannon(self, distribution):
        return -tf.multiply(1.0 / self.triplet_num, tf.tensordot(distribution, tf.log(distribution), 2))

    def network(self, x):
        weights = []
        fc1 = self.fc_layer(x, 256, "fc1")
        ac1 = tf.nn.relu(fc1)
        fc2 = self.fc_layer(ac1, 10, "fc2")
        ac2 = tf.nn.relu(fc2)
        fc3 = self.fc_layer(ac2, 256, "fc3")
        ac3 = tf.nn.relu(fc3)
        fc4 = self.fc_layer(ac3, 7, "fc4")
        # add a softmax to make it normalized
        return fc2

    def triplet_loss(self):
        lambda_penalty = tf.get_variable('lambda_penalty', [1, 1], dtype=tf.float32,
                                         initializer=tf.random_uniform_initializer)
        margin = 0.0000001
        # o1 is the anchor, o2 is the positive example and o3 is the negative example
        C = tf.constant(margin, name="C")
        eucd_p = tf.pow(tf.subtract(self.o1, self.o2), 2, "dist_pos")
        eucd_p = tf.reduce_sum(eucd_p, 1)
        eucd_n = tf.pow(tf.subtract(self.o1, self.o3), 2, 'dist_neg')
        eucd_n = tf.reduce_sum(eucd_n, 1)
        losses_no_margin = tf.subtract(eucd_p, eucd_n, 'losses_without_margin')
        losses = tf.add(losses_no_margin, C, name='losses')
        loss = tf.reduce_mean(tf.add(losses, tf.reduce_mean(
            tf.multiply(0.000000001, tf.add(self.entropy_shannon(self.o1), tf.add(self.entropy_shannon(self.o2), \
                                                                                  self.entropy_shannon(self.o3)))))))
        # loss = tf.reduce_mean(tf.maximum(losses, 0.0), name='loss')

        return loss

    def fc_layer(self, bottom, n_weight, name):
        assert len(bottom.get_shape()) == 2
        n_prev_weight = bottom.get_shape()[1]
        initer = tf.truncated_normal_initializer(stddev=0.01)
        W = tf.get_variable(name + 'W', dtype=tf.float32, shape=[n_prev_weight, n_weight], initializer=initer)
        b = tf.get_variable(name + 'b', dtype=tf.float32,
                            initializer=tf.constant(0.01, shape=[n_weight], dtype=tf.float32))
        fc = tf.nn.bias_add(tf.matmul(bottom, W), b)
        return fc
