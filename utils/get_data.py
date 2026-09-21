# Helper functions for downloading datasets

import os
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

MNIST_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
FASHION_MNIST_URLS = [
    "https://storage.googleapis.com/tensorflow/tf-keras-datasets/train-images-idx3-ubyte.gz",
    "https://storage.googleapis.com/tensorflow/tf-keras-datasets/train-labels-idx1-ubyte.gz",
    "https://storage.googleapis.com/tensorflow/tf-keras-datasets/t10k-images-idx3-ubyte.gz",
    "https://storage.googleapis.com/tensorflow/tf-keras-datasets/t10k-labels-idx1-ubyte.gz",
]
# CIFAR-10: the official source (https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz)
# downloads too slowly (~60 KB/s). This is not official, but it has the same MD5 hash
# as the official file (c58f30108f718f92721af3b95e74349a), so it is ok.
CIFAR_10_URL = "https://huggingface.co/datasets/liangnanying/cifar-10-python/resolve/main/cifar-10-python.tar.gz"

DATASET_URLS = {
    "mnist": [MNIST_URL],
    "fashion_mnist": FASHION_MNIST_URLS,
    "cifar10": [CIFAR_10_URL],
}

DEFAULT_DIR = str(Path(__file__).resolve().parent.parent / "data")


def _download(url: str,
              dest: str,
              verbose: bool = True,
              chunk_size: int = 1 << 20,
              **kwargs):
    """Stream a file from url to dest through a temporary .part file.
        Input:
            - url (str): link to download from
            - dest (str): destination path on disk
            - verbose (bool): whether to print download progress
            - chunk_size (int): number of bytes to read per chunk
            - **kwargs: additional keyword arguments for urllib.request.urlopen (e.g. timeout)
        Output:
            - None
    """
    tmp = dest + ".part"
    with urllib.request.urlopen(url, **kwargs) as response, open(tmp, "wb") as out:
        total = response.getheader("Content-Length")
        total = int(total) if total else None
        done = 0
        while chunk := response.read(chunk_size):
            out.write(chunk)
            done += len(chunk)
            if verbose and total:
                print(f"\r{done / total:7.1%}", end="", flush=True)

    if verbose:
        print()
        
    os.replace(tmp, dest)   


def get_dataset(dir: str,
                dataset_name: str,
                link: str | None = None,
                force_download=False,
                verbose=True,
                **kwargs):
    """Download a dataset to dir if it is not already downloaded.
        Input:
            - dir (str): directory to download the dataset to (files are placed in dir/<dataset_name>/, created if missing)
            - dataset_name (str): one of 'mnist', 'fashion_mnist', 'cifar10'
            - link (str): alternate base URL to download the dataset from (must end with '/', each file name is appended)
            - force_download (bool): whether to force download the dataset even if it already exists
            - verbose (bool): whether to print progress messages during download
            - **kwargs: additional keyword arguments to pass to the download function (e.g. timeout)
        Output:
            - paths (list[str]): local paths of the dataset files on disk
    """
    urls = DATASET_URLS.get(dataset_name)
    if urls is None:
        raise ValueError(f"Unknown dataset name {dataset_name!r}; choose one of {sorted(DATASET_URLS)}.")

    if link is not None:
        if not link.endswith("/"):
            raise ValueError("link must be a base URL ending with '/'.")
        urls = [link + os.path.basename(urlparse(url).path) for url in urls]

    os.makedirs(os.path.join(dir, dataset_name), exist_ok=True)

    paths = []
    for url in urls:
        path = os.path.join(dir, dataset_name, os.path.basename(urlparse(url).path))
        if not os.path.exists(path) or force_download:
            if verbose:
                print(f"{dataset_name}: downloading {url}")
            _download(url, path, verbose=verbose, **kwargs)
        paths.append(path)

    if verbose:
        print(f"{dataset_name}: {len(paths)} file(s) ready at {os.path.join(dir, dataset_name)}")

    return paths


def get_all_datasets(dir: str = DEFAULT_DIR,
                     force_download=False,
                     verbose=True,
                     **kwargs):
    """Download all datasets used in the course project.
        Input:
            - dir (str): directory to download the datasets to (each dataset gets its own subfolder inside it; default: repo-root/data regardless of where the script is run from; created if missing)
            - force_download (bool): whether to force download the datasets even if they already exist
            - verbose (bool): whether to print progress messages during download
            - **kwargs: additional keyword arguments to pass to each download (e.g. timeout)
        Output:
            - datasets (dict): maps each dataset name to the list of local paths of its files
    """
    datasets = {}
    for name in ("fashion_mnist", "mnist", "cifar10"):
        datasets[name] = get_dataset(dir, name,
                                     force_download=force_download,
                                     verbose=verbose,
                                     **kwargs)
        
    return datasets


if __name__ == "__main__":
    datasets = get_all_datasets()
    print("Datasets downloaded successfully.")
