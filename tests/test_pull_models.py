import unittest
from predict.pull_models import check_remote_file, download_remote_file

class TestPullModels(unittest.TestCase):
    def setUp(self):
        pass

    def test_check_remote_file(self):
        output = check_remote_file("CNN", "TT")
        print(output)

    def tearDown(self):
        pass


if __name__ == "__main__":
    unittest.main()
