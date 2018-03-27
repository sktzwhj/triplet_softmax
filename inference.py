import tensorflow as tf
import numpy as np
import tensorflow.contrib.bayesflow.entropy as entr

class siamese:

    # Create model
    def __init__(self):

        self.x1 = tf.placeholder(tf.float32, [None, 10])
        self.x2 = tf.placeholder(tf.float32, [None, 10])
        self.x3 = tf.placeholder(tf.float32, [None, 10])
        with tf.variable_scope("siamese") as scope:
            self.o1 = self.network(self.x1)
            scope.reuse_variables()
            self.o2 = self.network(self.x2)
            scope.reuse_variables()
            self.o3 = self.network(self.x3)

        self.y_ = tf.placeholder(tf.float32, [None])
        self.loss = self.triplet_loss()


    def network(self, x):
        weights = []
        fc1 = self.fc_layer(x, 256, "fc1")
        ac1 = tf.nn.relu(fc1)
        fc2 = self.fc_layer(ac1, 10, "fc2")
        ac2 = tf.nn.relu(fc2)
        fc3 = self.fc_layer(ac2, 256, "fc3")
        ac3 = tf.nn.relu(fc3)
        fc4 = self.fc_layer(ac3, 7, "fc4")
        #add a softmax to make it normalized
        return fc2


    def triplet_loss(self):
        margin = 0.00001
        #o1 is the anchor, o2 is the positive example and o3 is the negative example
        C = tf.constant(margin, name = "C")
        eucd_p = tf.pow(tf.subtract(self.o1, self.o2), 2, "dist_pos")
        eucd_p = tf.reduce_sum(eucd_p, 1)
        eucd_n = tf.pow(tf.subtract(self.o1, self.o3), 2, 'dist_neg')
        eucd_n = tf.reduce_sum(eucd_n, 1)
        losses_no_margin = tf.subtract(eucd_p, eucd_n, 'losses_without_margin')
        losses = tf.add(losses_no_margin, C, name='losses')
        loss = tf.reduce_mean(tf.maximum(losses, 0.0), name='loss')
        loss = tf.add(tf.add(loss, entr.entropy_shannon(self.o1)), tf.add(entr.entropy_shannon(self.o2), entr.entropy_shannon(self.o3)))
        return loss



    def fc_layer(self, bottom, n_weight, name):
        assert len(bottom.get_shape()) == 2
        n_prev_weight = bottom.get_shape()[1]
        initer = tf.truncated_normal_initializer(stddev=0.01)
        W = tf.get_variable(name+'W', dtype=tf.float32, shape=[n_prev_weight, n_weight], initializer=initer)
        b = tf.get_variable(name+'b', dtype=tf.float32, initializer=tf.constant(0.01, shape=[n_weight], dtype=tf.float32))
        fc = tf.nn.bias_add(tf.matmul(bottom, W), b)
        return fc

    def loss_with_spring(self):
        margin = self.alpha
        labels_t = self.y_
        labels_f = tf.subtract(1.0, self.y_, name="1-yi")          # labels_ = !labels;
        eucd2 = tf.pow(tf.subtract(self.o1, self.o2), 2)
        eucd2 = tf.reduce_sum(eucd2, 1)
        eucd = tf.sqrt(eucd2+1e-6, name="eucd")
        C = tf.constant(margin, name="C")
        # yi*||CNN(p1i)-CNN(p2i)||^2 + (1-yi)*max(0, C-||CNN(p1i)-CNN(p2i)||^2)
        pos = tf.multiply(labels_t, eucd2, name="yi_x_eucd2")
        # neg = tf.multiply(labels_f, tf.subtract(0.0,eucd2), name="yi_x_eucd2")
        # neg = tf.multiply(labels_f, tf.maximum(0.0, tf.subtract(C,eucd2)), name="Nyi_x_C-eucd_xx_2")
        neg = tf.multiply(labels_f, tf.pow(tf.maximum(tf.subtract(C, eucd), 0), 2), name="Nyi_x_C-eucd_xx_2")
        losses = tf.add(pos, neg, name="losses")
        loss = tf.reduce_mean(losses, name="loss")
        return loss




    def loss_with_step(self):
        margin = 5.0
        labels_t = self.y_
        labels_f = tf.subtract(1.0, self.y_, name="1-yi")          # labels_ = !labels;
        eucd2 = tf.pow(tf.subtract(self.o1, self.o2), 2)
        eucd2 = tf.reduce_sum(eucd2, 1)
        eucd = tf.sqrt(eucd2+1e-6, name="eucd")
        C = tf.constant(margin, name="C")
        pos = tf.multiply(labels_t, eucd, name="y_x_eucd")
        neg = tf.multiply(labels_f, tf.maximum(0.0, tf.subtract(C, eucd)), name="Ny_C-eucd")
        losses = tf.add(pos, neg, name="losses")
        loss = tf.reduce_mean(losses, name="loss")
        return loss
