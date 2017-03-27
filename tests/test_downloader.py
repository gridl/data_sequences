from data_sequences import downloader


def test_report_download_progress():
    downloader.report_download_progress(2, 10, 100)
