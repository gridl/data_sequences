import os


def makedirs(f_name):
    """same as os.makedirs(f_name, exists_ok=True) at python3"""
    if not os.path.exists(f_name):
        os.makedirs(f_name)
