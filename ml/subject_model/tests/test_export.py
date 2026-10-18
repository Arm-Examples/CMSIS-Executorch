"""Checks for export-investigation metric reporting (no Arm tools required)."""
import unittest
import torch
from src.export import metrics


class ExportMetricsTests(unittest.TestCase):
    def test_balanced_accuracy_with_unequal_class_counts(self):
        logits = torch.tensor([[2., 0.], [2., 0.], [2., 0.], [2., 0.]])
        labels = torch.tensor([0, 0, 0, 1])
        actual = metrics(logits, labels)
        self.assertEqual(actual['confusion_matrix'], [[3, 0], [1, 0]])
        self.assertEqual(actual['accuracy'], 0.75)
        self.assertEqual(actual['balanced_accuracy'], 0.5)


if __name__ == '__main__':
    unittest.main()
