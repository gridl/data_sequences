from __future__ import print_function, division

import numpy as np


class InvalidUsageError(Exception):
    pass


class DataSet:
    """Class to represent some dataset: train, validation, test, etc."""
    @property
    def num_examples(self):
        """Return qtty of examples in dataset"""
        raise NotImplementedError

    def next_batch(self, batch_size):
        """Return batch of required size of data, labels"""
        raise NotImplementedError

    @staticmethod
    def _shuffle_N_arrays(arrays):
        """Shuffle N numpy arrays with same indexes
        Args:
            arrays: list of numpy arrays
        Return:
            shuffled_arrays: list of numpy arrays
        """
        shapes = [np.asarray(ar).shape[0] for ar in arrays]
        if len(set(shapes)) != 1:
            msg = ("Shapes of arrays first dimension or list length should be "
                   "equal.")
            raise InvalidUsageError(msg)
        rand_indexes = np.random.permutation(arrays[0].shape[0])
        shuffled_arrays = []
        for array in arrays:
            if isinstance(array, np.ndarray):
                shuffled_arrays.append(array[rand_indexes])
            else:
                res = []
                for ind in rand_indexes:
                    res.append(array[ind])
                shuffled_arrays.append(res)
        return shuffled_arrays


class DataProvider:
    @property
    def data_shape(self):
        """Return shape as python list of one data entry"""
        raise NotImplementedError

    @property
    def n_classes(self):
        """Return `int` of num classes"""
        return self._n_classes

    @staticmethod
    def labels_to_one_hot(labels, n_classes):
        """Convert 1D array of labels to one hot representation
        
        Args:
            labels: 1D numpy array
        Return:
            new_labels: 2D numpy array
        """
        new_labels = np.zeros((labels.shape[0], n_classes))
        new_labels[range(labels.shape[0]), labels] = np.ones(labels.shape)
        return new_labels

    @staticmethod
    def labels_from_one_hot(labels):
        """Convert 2D array of labels to 1D class based representation
        
        Args:
            labels: 2D numpy array
        Return:
            new_labels: 1D numpy array
        """
        new_labels = np.argmax(labels, axis=1)
        return new_labels
