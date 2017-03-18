import os
import tempfile

import numpy as np

from data_sequences import arithmetic_two_numbers as atn


def test_BaseNumbersDataSet():
    max_inputs_length = 5
    fetched_size = 10
    dataset = atn.BaseNumbersDataSet()
    dataset.delimiter = 0
    dataset.max_inputs_length = max_inputs_length
    res = dataset.random_data(fetched_size)
    assert all(isinstance(a, np.ndarray) for a in res)
    assert res[0].shape == (fetched_size, max_inputs_length)
    assert res[1].shape == (fetched_size, max_inputs_length)
    assert res[2].shape == (fetched_size, max_inputs_length * 2 + 1)
    assert res[3].shape == (fetched_size, max_inputs_length + 1)
    # check that two numbers really summed
    n_1 = int(''.join(str(i) for i in res[0][0]))
    n_2 = int(''.join(str(i) for i in res[1][0]))
    assert n_1 + n_2 == int(''.join(str(i) for i in res[3][0]))
    # check that all works with string delimiter
    dataset.delimiter = 'some'
    res = dataset.random_data(fetched_size)
    assert isinstance(res[2], list)
    assert len(res[2]) == fetched_size
    assert len(res[2][0]) == (max_inputs_length * 2 + 1)


def test_BaseNumbersDataSet_one_hot():
    max_inputs_length = 5
    fetched_size = 4
    dataset = atn.BaseNumbersDataSet()
    dataset.delimiter = 0
    n_classes = 10
    dataset.max_inputs_length = max_inputs_length
    res = dataset.random_data(fetched_size, one_hot=True)
    assert all(isinstance(a, np.ndarray) for a in res)
    assert res[0].shape == (fetched_size, max_inputs_length, n_classes)
    assert res[1].shape == (fetched_size, max_inputs_length, n_classes)
    assert res[2].shape == (fetched_size, max_inputs_length * 2 + 1, n_classes)
    assert res[3].shape == (fetched_size, max_inputs_length + 1, n_classes)
    # check that two numbers really summed
    n_1 = int(''.join(str(i) for i in np.argmax(res[0][0], axis=1)))
    n_2 = int(''.join(str(i) for i in np.argmax(res[1][0], axis=1)))
    assert n_1 + n_2 == int(
        ''.join(str(i) for i in np.argmax(res[3][0], axis=1)))
    # check that all works with string delimiter
    dataset.delimiter = 'some'
    res = dataset.random_data(fetched_size, one_hot=True)
    assert isinstance(res[2], list)
    assert len(res[2]) == fetched_size
    assert len(res[2][0]) == (max_inputs_length * 2 + 1)
    assert len(res[2][0][0]) == n_classes


def test_RandomNumbersSumDataSet():
    max_inputs_length = 5
    batch_size = 10
    dataset = atn.RandomNumbersSumDataSet(max_inputs_length=max_inputs_length)
    assert dataset.num_examples is None
    batch_res = dataset.next_batch(10)
    assert all(res.shape[0] == batch_size for res in batch_res)


def test_TwoNumbersRandDataProvider():
    max_inputs_length = 5
    batch_size = 10
    data_provider = atn.TwoNumbersRandDataProvider(
        max_inputs_length=max_inputs_length)
    assert data_provider.data_shape == max_inputs_length
    assert data_provider.concatenated_data_shape == (max_inputs_length * 2 + 1)
    data_provider.train.next_batch(batch_size)
    data_provider.valid.next_batch(batch_size)
    batch_res = data_provider.test.next_batch(batch_size)
    assert all(res.shape[0] == batch_size for res in batch_res)


def test_SameNumbersSumDataSet_not_shuffled():
    max_inputs_length = 5
    dataset_size = 20
    batch_size = 5
    dataset = atn.SameNumbersSumDataSet(
        max_inputs_length=max_inputs_length,
        dataset_size=dataset_size)
    assert dataset.num_examples == dataset_size
    dataset.next_batch(batch_size)
    assert dataset._batch_counter == 1
    dataset.start_new_epoch()
    assert dataset._batch_counter == 0
    # simulate one epoch
    first_batch_first_epoch = None
    for i in range(dataset.num_examples // batch_size):
        res = dataset.next_batch(batch_size)
        if first_batch_first_epoch is None:
            first_batch_first_epoch = res
        assert dataset._batch_counter == i + 1
    # new epoch begins, batch counter should be renewed
    first_batch_second_epoch = dataset.next_batch(batch_size)
    assert dataset._batch_counter == 1
    # test that arrays were not shuffled
    for i in range(batch_size):
        assert all(a == b for a, b in zip(
            first_batch_first_epoch[0][i], first_batch_second_epoch[0][i]))


def test_SameNumbersSumDataSet_shuffled():
    max_inputs_length = 5
    dataset_size = 10
    batch_size = 5
    dataset = atn.SameNumbersSumDataSet(
        max_inputs_length=max_inputs_length,
        dataset_size=dataset_size,
        shuffle=True
    )
    for i in range(dataset.num_examples // batch_size):
        res_epoch_1 = dataset.next_batch(batch_size)

    for i in range(dataset.num_examples // batch_size):
        res_epoch_2 = dataset.next_batch(batch_size)
    results = []
    for i in range(batch_size):
        results.append(all(
            a == b for a, b in zip(res_epoch_1[0][i], res_epoch_2[0][i])))
    assert not all(results)


def test_TwoNumbersConstDataProvider_simple():
    max_inputs_length = 5
    train_size = 20
    valid_size = 10
    test_size = 10
    batch_size = 5
    data_provider = atn.TwoNumbersConstDataProvider(
        max_inputs_length=max_inputs_length,
        train_size=train_size,
        valid_size=valid_size,
        test_size=test_size)
    data_provider.train.next_batch(batch_size)
    data_provider.valid.next_batch(batch_size)
    data_provider.test.next_batch(batch_size)


def test_TwoNumbersConstDataProvider_cache(tmpdir):
    max_inputs_length = 5
    train_size = 20
    valid_size = 10
    test_size = 10
    batch_size = 5
    data_provider = atn.TwoNumbersConstDataProvider(
        max_inputs_length=max_inputs_length,
        train_size=train_size,
        valid_size=valid_size,
        test_size=test_size,
        shuffle=False,
        use_cache=True,
        cache_root_dir=str(tmpdir))
    train_batch_1 = data_provider.train.next_batch(batch_size)
    # this data provider should use cached data
    data_provider = atn.TwoNumbersConstDataProvider(
        max_inputs_length=max_inputs_length,
        train_size=train_size,
        valid_size=valid_size,
        test_size=test_size,
        shuffle=False,
        use_cache=True,
        cache_root_dir=str(tmpdir))
    train_batch_2 = data_provider.train.next_batch(batch_size)
    for i in range(batch_size):
        assert all(a == b for a, b in zip(
            train_batch_1[0][i], train_batch_2[0][i]))


def test_TwoNumbersConstDataProvider_cache_path():
    data_provider = atn.TwoNumbersConstDataProvider(
        max_inputs_length=5,
        train_size=10,
        valid_size=10,
        test_size=10,
        use_cache=True)
    path = data_provider.cache_dir_path
    check_path = os.path.join(
        tempfile.gettempdir(), 'arithmetic_two_numbers_dataset')
    assert path == check_path
