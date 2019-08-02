import os
import sys
import unittest

idx = os.getcwd().index("upbit_auto_trade")
PROJECT_HOME = os.getcwd()[:idx] + "upbit_auto_trade/"
sys.path.append(PROJECT_HOME)
os.chdir(PROJECT_HOME)

from predict.early_stopping import EarlyStopping


class TestEarlyStopping(unittest.TestCase):
    def setUp(self):
        self.es = EarlyStopping("CNN", "BSV")

    def test_push(self):
        output = self.es.push_models("./models/CNN/BSV_18_0.47_81.70_224_0.33.pt")
        print(output)

    def tearDown(self):
        pass


if __name__ == "__main__":
    unittest.main()
