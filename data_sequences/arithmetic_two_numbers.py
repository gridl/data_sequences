from __future__ import print_function, division
import os
import random
import tempfile

import numpy as np
from six.moves import cPickle

from .base_classes import DataSet, DataProvider
from .utils import makedirs, nd_array_to_one_hot


class BaseNumbersDataSet(DataSet):
    def random_data(self, size, one_hot=False):
        """Return arrays of random inputs, concatenated input and targets
        of requires size.
        Args:
            size: `int`, first dimension of generated data
            one_hot: `bool`, default=False,
                return inputs and labels one hot encoder or not
        Returns:
            inputs_1: 2D numpy array, shape (size, max_inputs_length)
            inputs_2: 2D numpy array, shape (size, max_inputs_length)
            inputs_concat: 2D numpy array or list of lists,
                shape (size, max_inputs_length * 2 + 1)
            targets: 2D numpy array, shape (size, max_inputs_length + 1)
        """
        n_classes = 10
        inputs_1 = np.zeros((size, self.max_inputs_length), dtype='uint8')
        inputs_2 = np.zeros((size, self.max_inputs_length), dtype='uint8')
        inputs_concat = []
        targets = np.zeros((size, self.max_inputs_length + 1), dtype='uint8')
        for i in range(size):
            inp_1 = random.randint(1, 10 ** self.max_inputs_length)
            inp_2 = random.randint(1, 10 ** self.max_inputs_length)
            target = inp_1 + inp_2
            inp_1_padded = [
                int(i) for i in str(inp_1).zfill(self.max_inputs_length)]
            inp_2_padded = [
                int(i) for i in str(inp_2).zfill(self.max_inputs_length)]
            target_padded = [
                int(i) for i in str(target).zfill(self.max_inputs_length + 1)]
            conc_inputs = inp_1_padded + [self.delimiter] + inp_2_padded
            inputs_1[i] = np.asarray(inp_1_padded, dtype='uint8')
            inputs_2[i] = np.asarray(inp_2_padded, dtype='uint8')
            targets[i] = np.asarray(target_padded, dtype='uint8')
            inputs_concat.append(conc_inputs)
        if isinstance(self.delimiter, int):
            inputs_concat = np.asarray(inputs_concat, dtype='uint8')
        if one_hot:
            inputs_1 = nd_array_to_one_hot(inputs_1, n_classes)
            inputs_2 = nd_array_to_one_hot(inputs_2, n_classes)
            targets = nd_array_to_one_hot(targets, n_classes)
            if isinstance(self.delimiter, int):
                delimiter_ar = np.tile(np.array(
                    self.delimiter, dtype='uint8'), (size, 1, n_classes))
                inputs_concat = np.concatenate(
                    (inputs_1, delimiter_ar, inputs_2), axis=1)
            else:
                delimiter_list = [self.delimiter] * n_classes
                inputs_concat = [
                    a.tolist() + [delimiter_list] + b.tolist() for a, b in zip(
                        inputs_1, inputs_2)]
        return inputs_1, inputs_2, inputs_concat, targets


class RandomNumbersSumDataSet(BaseNumbersDataSet):
    def __init__(self, max_inputs_length, delimiter=0, one_hot=False):
        """
        Args:
            max_inputs_length: `int`, max length of inputs.
            delimiter: any type. Delimiter for concatenated inputs.
            one_hot: `bool`, default=False,
                return inputs and labels one hot encoder or not
        """
        self.max_inputs_length = max_inputs_length
        self.delimiter = delimiter
        self.one_hot = one_hot

    @property
    def num_examples(self):
        return None

    def next_batch(self, batch_size):
        """Return batch of required size"""
        return self.random_data(batch_size, self.one_hot)


class SameNumbersSumDataSet(BaseNumbersDataSet):
    def __init__(self, max_inputs_length, dataset_size, shuffle=False,
                 delimiter=0, dataset_all_data=None, one_hot=False):
        """
        Args:
            max_inputs_length: `int`, max length of inputs.
            dataset_size: `int`, qtty of entries in dataset.
            shuffle: `bool`, default=False.
                Should generated numbers be shuffled every epoch or not.
            delimiter: any type. Delimiter for concatenated inputs.
            dataset_all_data: `tuple` of previously generated data
            one_hot: `bool`, default=False,
                return inputs and labels one hot encoder or not
        """
        self.max_inputs_length = max_inputs_length
        self.shuffle = shuffle
        self.dataset_size = dataset_size
        self.delimiter = delimiter
        self.dataset_all_data = dataset_all_data
        self.one_hot = one_hot
        if self.dataset_all_data is None:
            self.dataset_all_data = self.random_data(
                dataset_size, one_hot=one_hot)
        self.start_new_epoch()

    @property
    def num_examples(self):
        return self.dataset_size

    def next_batch(self, batch_size):
        start = self._batch_counter * batch_size
        end = (self._batch_counter + 1) * batch_size
        self._batch_counter += 1
        slices = []
        for part in self.dataset_all_data:
            slices.append(part[start:end])
        if slices[0].shape[0] != batch_size:
            self.start_new_epoch()
            return self.next_batch(batch_size)
        return slices

    def start_new_epoch(self):
        self._batch_counter = 0
        if self.shuffle:
            self.dataset_all_data = self._shuffle_N_arrays(
                self.dataset_all_data)


class TwoNumbersRandDataProvider(DataProvider):
    """Class representing data provider for two numbers addition task.
    Each call numbers generated randomly
    As inputs it will return two numbers as array left padded with zeros to
    required length, and as target sum of this numbers. Target length will
    be larger by 1 then inputs length(because of possible sum more than 10).
    Also concatenated inputs with some delimiter <del>(default to 0) are
    available.
    Example:
        input_1:        [0, 0, 1, 2, 3]
        input_2:        [0, 0, 9, 0, 1]
        target:      [0, 0, 1, 0, 2, 4]  # 123 + 901 = 1024
        conc_inputs: [0, 0, 1, 2, 3, <del>, 0, 0, 9, 0, 1]
    """

    def __init__(self, max_inputs_length, delimiter=0):
        """
        Args:
            max_inputs_length: `int`, max length of inputs
            delimiter: any type. Delimiter for concatenated inputs.
        """
        self.max_inputs_length = max_inputs_length
        self.train = self.valid = self.test = RandomNumbersSumDataSet(
            max_inputs_length=max_inputs_length,
            delimiter=delimiter)

    @property
    def data_shape(self):
        return self.max_inputs_length

    @property
    def concatenated_data_shape(self):
        return self.max_inputs_length * 2 + 1


class TwoNumbersConstDataProvider(TwoNumbersRandDataProvider):
    """Class representing data provider for two numbers addition task.
    Generated numbers are same for all epochs.
    As inputs it will return two numbers as array left padded with zeros to
    required length, and as target sum of this numbers. Target length will
    be larger by 1 then inputs length(because of possible sum more than 10).
    Also concatenated inputs with some delimiter <del>(default to 0) are
    available.
    Example:
        input_1:        [0, 0, 1, 2, 3]
        input_2:        [0, 0, 9, 0, 1]
        target:      [0, 0, 1, 0, 2, 4]  # 123 + 901 = 1024
        conc_inputs: [0, 0, 1, 2, 3, <del>, 0, 0, 9, 0, 1]
    """

    def __init__(self, max_inputs_length,
                 train_size, valid_size, test_size,
                 delimiter=0, shuffle=False, use_cache=False,
                 cache_root_dir=None, one_hot=False):
        """
        Args:
            max_inputs_length: `int`, max length of inputs.
            train_size: `int`, size of train dataset.
            valid_size: `int`, size of validation dataset.
            test_size: `int`, size of train dataset.
            delimiter: any type. Delimiter for concatenated inputs.
            shuffle: `bool`, default=False.
                Should generated numbers be shuffled every epoch or not.
            use_cache: `bool`, default=False. Save generated data in cache
                and use it for future runs with same settings.
            cache_root_dir: `str`, where cached data should be stored.
                default to os based tmp dir.
            one_hot: `bool`, default=False,
                return inputs and labels one hot encoder or not
        """
        self.max_inputs_length = max_inputs_length
        self.train_size = train_size
        self.valid_size = valid_size
        self.test_size = test_size
        self.delimiter = delimiter
        self.cache_root_dir = cache_root_dir
        self.one_hot = one_hot

        if use_cache:
            prev_data, cache_was_loaded = self._get_prev_data_if_exist()
        else:
            prev_data = self.blank_prev_data
            cache_was_loaded = False
        for d_name in ['train', 'valid', 'test']:
            dataset = SameNumbersSumDataSet(
                max_inputs_length,
                dataset_size=getattr(self, '%s_size' % d_name),
                shuffle=shuffle,
                delimiter=delimiter,
                dataset_all_data=prev_data[d_name],
                one_hot=one_hot,
            )
            setattr(self, d_name, dataset)
            if use_cache and not cache_was_loaded:
                prev_data[d_name] = dataset.dataset_all_data
        if use_cache and not cache_was_loaded:
            self._save_cache_data(prev_data)

    def _get_prev_data_if_exist(self):
        try:
            with open(self.cache_pickle_path, 'rb') as f:
                prev_data = cPickle.load(f)
                cache_was_loaded = True
        except FileNotFoundError:
            prev_data = self.blank_prev_data
            cache_was_loaded = False
        return prev_data, cache_was_loaded

    def _save_cache_data(self, new_data):
        with open(self.cache_pickle_path, 'wb') as f:
            cPickle.dump(new_data, f)

    @property
    def blank_prev_data(self):
        blank_dict = {
            'train': None,
            'valid': None,
            'test': None
        }
        return blank_dict

    @property
    def cache_dir_path(self):
        if not hasattr(self, '_cache_dir_path'):
            if self.cache_root_dir is None:
                self.cache_root_dir = tempfile.gettempdir()
            self._cache_dir_path = os.path.join(
                self.cache_root_dir, 'arithmetic_two_numbers_dataset')
            makedirs(self._cache_dir_path)
        return self._cache_dir_path

    @property
    def cache_pickle_path(self):
        f_name = (
            'max_inputs_length_{}_train_size_{}_valid_size_{}_'
            'test_size_{}_delimiter_{}_one_hot_{}.pkl'.format(
                self.max_inputs_length,
                self.train_size,
                self.valid_size,
                self.test_size,
                self.delimiter,
                self.one_hot))
        return os.path.join(self.cache_dir_path, f_name)
