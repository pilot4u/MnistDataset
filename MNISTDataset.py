import gzip

import numpy as np
import urllib.request, re
import os
from torch.utils.data import Dataset

conf = {"dataset_name": "fasion_mnist_2nd", "layout": [28, 28], "max_images_to_load": 30,
        "classes": ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag",
                    "Ankle boot"], "content": {
        "train": {
            "data": "https://beyondminds-test.s3.us-east-2.amazonaws.com/fasion_mnist/28X28/train-images-idx3-ubyte.gz",
            "labels": "https://beyondminds-test.s3.us-east-2.amazonaws.com/fasion_mnist/28X28/train-labels-idx1-ubyte.gz"},
        "test": {
            "data": "https://beyondminds-test.s3.us-east-2.amazonaws.com/fasion_mnist/28X28/t10k-images-idx3-ubyte.gz",
            "labels": "https://beyondminds-test.s3.us-east-2.amazonaws.com/fasion_mnist/28X28/t10k-labels-idx1-ubyte.gz"}}}

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(ROOT_DIR, 'data')

path_getter = re.compile(r'(?:.+\/)(.+)')


def get_file_from_uri(uri):
    return path_getter.findall(uri)[-1]


def get_file(uri):
    file_path = os.path.join(DATA_PATH,get_file_from_uri(uri))
    if not os.path.isfile(file_path):
        if not os.path.exists(DATA_PATH):
            os.makedirs(DATA_PATH)
        urllib.request.urlretrieve(uri, file_path)
    return file_path


class MNISTFashionTDataset(Dataset):
    def __init__(self, purpose="train"):
        if purpose not in conf['content']:
            raise Exception('purpase should be "train" or "test"')

        uris = conf['content'][purpose]
        data_path = get_file(uris['data'])
        labels_path = get_file(uris['labels'])
        with gzip.open(labels_path, 'rb') as lbpath:
            self.x_data = np.frombuffer(lbpath.read(), dtype=np.uint8,
                                        offset=8)

        self.len = len(self.x_data)

        with gzip.open(data_path, 'rb') as imgpath:
            self.y_data = np.frombuffer(imgpath.read(), dtype=np.uint8,
                                        offset=16).reshape(self.len, np.prod(conf['layout']))

    def __getitem__(self, index):
        return (self.x_data[index], self.y_data[index])

    def __len__(self):
        return self.len
