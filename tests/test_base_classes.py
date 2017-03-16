import numpy as np
import pytest

from data_sequences import base_classes


def test_dataset_num_examples():
    dataset = base_classes.DataSet()
    with pytest.raises(NotImplementedError):
        dataset.num_examples


def test_dataset_next_batch():
    dataset = base_classes.DataSet()
    with pytest.raises(NotImplementedError):
        dataset.next_batch(20)


def test_dataset_shuffle_N_arrays():
    dataset = base_classes.DataSet()
    test_arrays_1 = [np.arange(10)]
    shuffled_1 = dataset._shuffle_N_arrays(test_arrays_1)
    assert not all(np.all(a1 == a2) for a1, a2 in zip(test_arrays_1, shuffled_1))
    test_arrays_2 = [np.arange(10), np.arange(20)]
    shuffled_2 = dataset._shuffle_N_arrays(test_arrays_2)
    assert not all(np.all(a1 == a2) for a1, a2 in zip(test_arrays_2, shuffled_2))


def test_data_provider_from_to_one_hot():
    n_classes = 5
    labels_one_hot = np.array([
        [0, 0, 0, 1, 0],
        [1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1]
    ])
    labels_by_classes = np.array([3, 0, 2, 4])
    prov = base_classes.DataProvider()

    labels_one_hot_prov = prov.labels_to_one_hot(labels_by_classes, n_classes)
    assert np.all(labels_one_hot == labels_one_hot_prov)
    labels_by_classes_prov = prov.labels_from_one_hot(labels_one_hot)
    assert np.all(labels_by_classes == labels_by_classes_prov)
    # test multiply conversion
    labels_by_classes_prov_2 = prov.labels_from_one_hot(
        prov.labels_to_one_hot(labels_by_classes, n_classes))
    assert np.all(labels_by_classes == labels_by_classes_prov_2)


def test_data_provider_n_classes():
    prov = base_classes.DataProvider()
    with pytest.raises(AttributeError):
        prov.n_classes

    prov._n_classes = 5
    assert prov.n_classes == 5

    prov._n_classes = 10
    assert prov.n_classes == 10


def test_data_provider_data_shape():
    prov = base_classes.DataProvider()
    with pytest.raises(NotImplementedError):
        prov.data_shape
