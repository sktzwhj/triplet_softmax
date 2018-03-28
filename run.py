from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from builtins import input

import tensorflow as tf
import numpy as np
import os
import inference
import MNIST_softmax_data
import CIFAR_softmax_data

mnist_softmax = CIFAR_softmax_data.DataSets()
sess = tf.InteractiveSession()
# setup siamese network
siamese = inference.siamese()
train_step = tf.train.RMSPropOptimizer(0.01).minimize(siamese.loss)
saver = tf.train.Saver()
tf.initialize_all_variables().run()

load = False
model_ckpt = './model.meta'
if os.path.isfile(model_ckpt):
    input_var = None
    while input_var not in ['yes', 'no']:
        input_var = input("We found model files. Do you want to load it and continue training [yes/no]?")
    if input_var == 'yes':
        load = True

# start training
if load: saver.restore(sess, './model')

for step in range(100001):
    batchsize = 512
    batch_x, batch_y = mnist_softmax.train.next_batch(batchsize)
    embeddings = np.array(sess.run([siamese.o1], feed_dict={siamese.x1: batch_x})[0])
    embedding_labels = batch_y
    # print('embeddings shape:', embeddings.shape)
    # print('embedding labels shape:', embedding_labels.shape)
    triplets = []
    selected = set()
    alpha = 0.0000001
    # different number of triplets might be selected for different mini-batches.
    num_of_triplets = 0
    # choose the triplets in each mini-batch, the embeddings are provided by the  calls
    # i, j, k are the index of anchor, postive and negative, respectively
    for i in range(batchsize):
        distance_to_anchor = np.sum(np.square(embeddings[i] - embeddings), axis=1)
        pos_indices = np.where(np.equal(embedding_labels, embedding_labels[i]))[0]
        neg_indices = np.where(np.not_equal(embedding_labels, embedding_labels[i]))[0]
        for j in pos_indices:
            distance_to_anchor[np.isinf(distance_to_anchor) == True] = 0
            distance_to_anchor[np.isnan(distance_to_anchor) == True] = 0
            all_neg = np.where(distance_to_anchor[neg_indices] - distance_to_anchor[j] < alpha)[0]
            num_of_all_neg = all_neg.shape[0]
            if num_of_all_neg > 0:
                rnd_idx = np.random.randint(num_of_all_neg)
                n_id = neg_indices[rnd_idx]
                triplets.append([batch_x[i], batch_x[j], batch_x[n_id]])
                num_of_triplets += 1
    if num_of_triplets == 0:
        triplets.append([batch_x[pos_indices[np.random.randint(len(pos_indices))]],
                         batch_x[pos_indices[np.random.randint(len(pos_indices))]], \
                         batch_x[neg_indices[np.random.randint(len(neg_indices))]]])
    np.random.shuffle(triplets)
    triplets_anchors = np.array([t[0] for t in triplets])
    triplets_pos = np.array([t[1] for t in triplets])
    triplets_neg = np.array([t[2] for t in triplets])
    # print('the num of selected triplets is ', num_of_triplets)

    _, loss_v = sess.run([train_step, siamese.loss], feed_dict={
        siamese.x1: triplets_anchors,
        siamese.x2: triplets_pos,
        siamese.x3: triplets_neg,
        siamese.y_: batch_y,
        siamese.triplet_num: np.array(num_of_triplets).reshape(-1, 1)
    })

    if np.isnan(loss_v):
        print('Model diverged with loss = NaN')
        quit()

    if step % 10 == 0:
        print('step', step, 'loss', loss_v)

    if step % 50 == 0 and step > 0:
        saver.save(sess, './model')
        embed = siamese.o1.eval({siamese.x1: mnist_softmax.train_x})
        embed_test = siamese.o1.eval({siamese.x1: mnist_softmax.test_x})
        embed.tofile('embed.txt')
        embed[np.isinf(embed) == True] = 0
        embed[np.isnan(embed) == True] = 0

        from sklearn.neighbors.classification import *

        neigh = KNeighborsClassifier(n_neighbors=1)

        neigh.fit(embed, mnist_softmax.train_y)
        cnt = 0
        train_data = np.array(mnist_softmax.train_x)
        train_labels = np.array(mnist_softmax.train_y)
        test_data = np.array(mnist_softmax.test_x)
        for i in range(10000):
            predicted_label = neigh.predict(embed_test[i].reshape(1, -1))[0]
            if np.argmax(mnist_softmax.test_x[i]) == predicted_label:
                # print train_labels[i], predicted_label
                cnt += 1
        print(cnt)
