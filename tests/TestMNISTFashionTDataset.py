import unittest
from MNISTDataset import MNISTFashionTDataset


class MyTestCase(unittest.TestCase):
    def test_download_train_len(self):
        ds = MNISTFashionTDataset()
        item = ds.__getitem__(1)
        len = ds.__len__()
        self.assertEqual(len, 60000)


if __name__ == '__main__':
    unittest.main()
