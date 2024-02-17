
import os

def ensure_dir_single(path):
    """
    Create directory if not existing already

    Does not create parent directories
    """
    if not os.path.exists(path):
        os.mkdir(path)


def ensure_dir(path):
    """
    Create directory if not existing already

    Will create parent directories as needed
    """
    if not os.path.exists(path):
        os.makedirs(path)
