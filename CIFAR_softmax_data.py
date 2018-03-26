import numpy as np
import cPickle
import gzip
import numpy
import collections

def extract_labels(filename, num_images):
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = numpy.frombuffer(buf, dtype=numpy.uint8).astype(numpy.int64)
    return labels


def get_transformed_data(transformed_data_file, batch_size):
    transformed_data_raw = cPickle.load(open(transformed_data_file, 'r'))
    transformed_data_list = []
    for i in range(len(transformed_data_raw)):
        for j in range(batch_size):
            transformed_data_list.append(transformed_data_raw[i][j])
    return transformed_data_list


def get_transformed_label(transformed_data_file, batch_size):
    transformed_data_raw = cPickle.load(open(transformed_data_file, 'r'))
    transformed_data_list = []
    for i in range(len(transformed_data_raw)):
        for j in range(batch_size):
            # print(transformed_data_raw[i][j],'  ',train_labels[i*batch_size+j])
            transformed_data_list.append(transformed_data_raw[i][j].tolist().index(max(transformed_data_raw[i][j])))
    return transformed_data_list


def softmax(z):
    return np.divide(np.exp(z), np.sum(np.exp(z)))

CIFAR_DIR = '/data/wu061/cifar10/'
#CIFAR_DIR = '/home/wu061/bracewell_data/cifar10/'


def get_data_labels():
    transformed_dir = ''
    high_cifar = True
    if high_cifar:
        transformed_dir = 'highest_accuracy_transformed/'
    else:
        transformed_dir = 'higher_accuracy_transformed/'

    transformed_train_label = get_transformed_label(CIFAR_DIR + transformed_dir + 'logits_all_training_data', 100)
    transformed_train_data = get_transformed_data(CIFAR_DIR + transformed_dir + 'logits_all_training_data', 100)
    transformed_test_label = get_transformed_data(CIFAR_DIR + transformed_dir + 'labels_list_all_testing_data', 100)
    transformed_test_data = get_transformed_data(CIFAR_DIR + transformed_dir + 'logits_all_testing_data', 100)


    true_train_labels = get_transformed_data(CIFAR_DIR + transformed_dir + 'labels_list_all_training_data', 100)
    if False:
        for i in range(50000):
            transformed_train_data[i] = softmax(transformed_train_data[i])
        for j in range(10000):
            transformed_test_data[j] = softmax(transformed_test_data[j])
    return transformed_train_data,transformed_train_label,transformed_test_data,transformed_test_label,true_train_labels


train_data, train_labels, test_data, test_labels, _ = get_data_labels()
distance_rank = cPickle.load(open('./CIFAR_distance_matrix/different_label_index_order_CIFAR_91'))
svm_boundary_order = cPickle.load(open('./confidence_rank_list/CIFAR_91_ranklist_svm_multicore'))
from collections import defaultdict


class DataSet:
    def __init__(self, train_x, train_y):
        self._index = 0
        self.train_x = np.array(train_x)
        self.train_y = np.array(train_y)
        self._num_examples = len(train_x)
        self._epochs_completed = 0
        self._index_in_epoch = 0

    #def next_batch(self, batch_size):
    #    self._index = (self._index + batch_size)%(55000-batch_size-1)
    #    return np.array(self.train_x[self._index : self._index + batch_size]), np.array(self.train_y[self._index : self._index + batch_size])

    def next_batch_triplet_loss(self, batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1

            #print self._num_examples
            # Shuffle the data

            perm = numpy.arange(self._num_examples)
            numpy.random.shuffle(perm)
            self.train_x = self.train_x[perm]
            self.train_y = self.train_y[perm]

            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self.train_x[np.arange(start, end)], self.train_y[np.arange(start, end)]

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""

        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1

            #print self._num_examples
            # Shuffle the data

            perm = numpy.arange(self._num_examples)
            numpy.random.shuffle(perm)
            self.train_x = self.train_x[perm]
            self.train_y = self.train_y[perm]

            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        #print np.array(self.train_x[start:end]).shape, np.array(self.train_y[start:end]).shape
        return np.array(self.train_x[start:end]), np.array(self.train_y[start:end])




class DataSets:
    def __init__(self):
        self._index = 0

        self.train_x, self.train_y, self.test_x, self.test_y, _ = get_data_labels()

        self.train = DataSet(self.train_x, self.train_y)
        self.test = DataSet(self.test_x, self.test_y)

from sklearn.neighbors import KNeighborsClassifier

if __name__ == "__main__":
    train_data, train_labels, test_data, test_labels, true_train_labels = get_data_labels()
    neigh = KNeighborsClassifier(n_neighbors=1)
    neigh.fit(train_data, train_labels)
    cnt = 0
    train_data = np.array(train_data)
    train_labels = np.array(train_labels)
    test_data = np.array(test_data)
    for i in range(10000):
        predicted_label = neigh.predict(test_data[i].reshape(1,-1))[0]
        if np.argmax(test_data[i]) == predicted_label:
            cnt += 1
    print cnt
