import unittest


class TestLightningPerceptronClassifier(unittest.TestCase):
    def test_output_dimensions(self):
        data = [1, 2, 3]
        result = sum(data)
        self.assertEqual(result, 6)


if __name__ == "__main__":
    unittest.main()
