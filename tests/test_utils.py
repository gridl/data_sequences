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
