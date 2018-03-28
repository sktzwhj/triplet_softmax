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


def get_data_labels_mnist_hidden():
    train_labels_filename = './MNIST_data/train-labels-idx1-ubyte.gz'
    test_labels_filename = './MNIST_data/t10k-labels-idx1-ubyte.gz'
    transformed_train_label = get_transformed_label(MNIST_DIR + 'drift_full/logits_all_training_data', 50)
    transformed_train_data = get_transformed_data('/home/wu061/bracewell_data/image_embedding_mnist_training', 50)
    transformed_test_data = get_transformed_data('/home/wu061/bracewell_data/image_embedding_mnist_testing', 50)
    transformed_test_label = extract_labels(test_labels_filename, 10000)
    true_labels = extract_labels(train_labels_filename, 60000)
    transformed_test_confidence = get_transformed_data(MNIST_DIR + 'drift_full/logits_all_testing_data', 50)
    model_test_label = [np.argmax(k) for k in transformed_test_confidence]
    import sklearn
    sklearn.preprocessing.normalize(transformed_train_data, norm='l2')
    sklearn.preprocessing.normalize(transformed_test_data, norm='l2')

    return transformed_train_data, transformed_train_label, transformed_test_data, transformed_test_label, true_labels, model_test_label


def get_data_labels(N=60000):
    train_labels_filename = './MNIST_data/train-labels-idx1-ubyte.gz'
    test_labels_filename = './MNIST_data/t10k-labels-idx1-ubyte.gz'
    transformed_train_label = get_transformed_label(MNIST_DIR + 'drift_full/logits_all_training_data', 50)
    transformed_train_data = get_transformed_data(MNIST_DIR + 'drift_full/logits_all_training_data', 50)
    transformed_test_data = get_transformed_data(MNIST_DIR + 'drift_full/logits_all_testing_data', 50)
    transformed_test_label = extract_labels(test_labels_filename, 10000)
    true_labels = extract_labels(train_labels_filename, 60000)
    if True:
        for i in range(60000):
            transformed_train_data[i] = softmax(transformed_train_data[i])
        for j in range(10000):
            transformed_test_data[j] = softmax(transformed_test_data[j])
    return transformed_train_data[:N], transformed_train_label[
                                       :N], transformed_test_data, transformed_test_label, true_labels


def ambiguous_pair_mix():
    distance_rank = cPickle.load(open('./MNIST_distance_matrix/top6000_distance_rank'))
    svm_boundary_order = cPickle.load(open('./confidence_rank_list/MNIST_ranklist_svm_after_softmax'))
    train_data, train_labels, test_data, test_labels, _ = get_data_labels(N=60000)
    train_data_reorder = []
    train_labels_reorder = []
    train_data_set = set()

    # index = svm_boundary_order[0]
    # train_data_set.add(index)
    for index in svm_boundary_order:
        for i in range(len(distance_rank[index])):
            if i not in train_data_set:
                train_data_reorder.append(train_data[i])
                train_labels_reorder.append(train_labels[i])
                train_data_set.add(i)

    while False:
        if len(train_labels_reorder) % 10 == 0:
            print len(train_labels_reorder), 100
        if len(train_labels_reorder) > 128 and len(train_labels_reorder) > 128:
            break
        for i in distance_rank[index]:
            if train_labels[i] != train_labels[index] and (i not in train_data_set):
                train_data_reorder.append(train_data[i])
                train_labels_reorder.append(train_labels[i])
                train_data_set.add(i)
                index = i
                break
    return np.array(train_data_reorder), np.array(train_labels_reorder), np.array(test_data), np.array(test_labels)


class DataSet:
    def __init__(self, train_x, train_y):
        self._index = 0
        self.train_x = np.array(train_x)
        self.train_y = np.array(train_y)
        self._num_examples = len(train_x)
        self._epochs_completed = 0
        self._index_in_epoch = 0

    # def next_batch(self, batch_size):
    #    self._index = (self._index + batch_size)%(55000-batch_size-1)
    #    return np.array(self.train_x[self._index : self._index + batch_size]), np.array(self.train_y[self._index : self._index + batch_size])


    def next_batch(self, batch_size, ambuguity=False):
        """Return the next `batch_size` examples from this data set."""

        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            if not ambuguity:
                perm = numpy.arange(self._num_examples)
                numpy.random.shuffle(perm)
                self.train_x = self.train_x[perm]
                self.train_y = self.train_y[perm]

            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return np.array(self.train_x[start:end]), np.array(self.train_y[start:end])


class DataSets:
    def __init__(self, ambiguity=False):
        self._index = 0
        if not ambiguity:
            self.train_x, self.train_y, self.test_x, self.test_y, _ = get_data_labels()
        else:
            self.train_x, self.train_y, self.test_x, self.test_y = ambiguous_pair_mix()
        self.train = DataSet(self.train_x, self.train_y)
        self.test = DataSet(self.test_x, self.test_y)


from sklearn.neighbors import KNeighborsClassifier

MNIST_DIR = '/data/wu061/mnist/'
if __name__ == "__main__":
    train_data, train_labels, test_data, test_labels, true_train_labels = get_data_labels()
    # from pylmnn.lmnn import LargeMarginNearestNeighbor as LMNN
    neigh = KNeighborsClassifier(n_neighbors=1)
    # clf = LMNN(n_neighbors=1, max_iter=10, n_features_out=10)

    neigh.fit(train_data, train_labels)
    cnt = 0
    train_data = np.array(train_data)
    train_labels = np.array(train_labels)
    test_data = np.array(test_data)
    for i in range(10000):
        predicted_label = neigh.predict(test_data[i].reshape(1, -1))[0]
        if test_labels[i] == predicted_label:
            # print train_labels[i], predicted_label
            cnt += 1
    print cnt
