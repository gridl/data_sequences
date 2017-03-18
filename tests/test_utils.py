import numpy as np

from data_sequences import utils


def test_downloader_makedirs(tmpdir):
    root_path = tmpdir.mkdir("root_dir")
    new_path = root_path.join("new_folder")
    # check that root dir now empty
    assert len(root_path.listdir()) == 0
    # check that new folder was created
    utils.makedirs(str(new_path))
    assert len(root_path.listdir()) == 1
    # check that no any new folders were created and no any exceptions
    utils.makedirs(str(new_path))
    assert len(root_path.listdir()) == 1


def test_nd_array_to_from_one_hot():
    n_classes = 10
    arr = np.tile(np.arange(n_classes), (5, 2))
    # convert to one hot
    converted = utils.nd_array_to_one_hot(arr, n_classes)
    assert converted.shape[-1] == n_classes
    assert len(converted.shape) == len(arr.shape) + 1
    for i in range(n_classes):
        assert sum(converted[0][i]) == 1
    # convert back from one hot
    converted_back = utils.nd_array_from_one_hot(converted)
    assert converted_back.shape == arr.shape
    assert np.all(arr == converted_back)
