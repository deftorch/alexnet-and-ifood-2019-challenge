import unittest
import torch
from src.models.alexnet import get_model

class TestAlexNet(unittest.TestCase):
    def test_alexnet_baseline_shape(self):
        model = get_model('alexnet_baseline', num_classes=251)
        input_tensor = torch.randn(2, 3, 224, 224)
        output = model(input_tensor)
        self.assertEqual(output.shape, (2, 251))

    def test_alexnet_mod1_shape(self):
        model = get_model('alexnet_mod1', num_classes=251)
        input_tensor = torch.randn(2, 3, 224, 224)
        output = model(input_tensor)
        self.assertEqual(output.shape, (2, 251))

    def test_alexnet_combined_shape(self):
        model = get_model('alexnet_combined', num_classes=251)
        input_tensor = torch.randn(2, 3, 224, 224)
        output = model(input_tensor)
        self.assertEqual(output.shape, (2, 251))

if __name__ == '__main__':
    unittest.main()
