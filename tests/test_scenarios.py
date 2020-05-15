import unittest

from src.helper import Helper


class TestScenarios(unittest.TestCase):

    def test_convnet(self):
        self.assertTrue(Helper().scenarios_works("convnet"))

    def test_mlp(self):
        self.assertTrue(Helper().scenarios_works("mlp"))

    def test_resnet(self):
        self.assertTrue(Helper().scenarios_works("resnet"))

    def test_rnn(self):
        self.assertTrue(Helper().scenarios_works("rnn"))


if __name__ == '__main__':
    unittest.main()
