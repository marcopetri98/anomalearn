import unittest

from anomalearn.utils import true_positive_rate, true_negative_rate, \
    binary_confusion_matrix


class TestMetricsFunctions(unittest.TestCase):
    def test_true_positive_rate(self):
        tpr = true_positive_rate([0, 0, 1, 0, 1, 0, 1], [0, 0, 1, 0, 1, 0, 1])
        self.assertEqual(1, tpr)
        
        tpr = true_positive_rate([0, 0, 1, 0, 1, 0, 1], [0, 0, 1, 0, 1, 0, 0])
        self.assertEqual(2/3, tpr)
        
        tpr = true_positive_rate([0, 0, 1, 0, 1, 0, 1], [0, 0, 1, 0, 0, 0, 1])
        self.assertEqual(2/3, tpr)
        
        tpr = true_positive_rate([0, 0, 1, 0, 1, 0, 1], [0, 0, 1, 0, 0, 0, 0])
        self.assertEqual(1/3, tpr)
        
        tpr = true_positive_rate([0, 0, 1, 0, 1, 0, 1], [0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(0, tpr)
    
    def test_true_negative_rate(self):
        tnr = true_negative_rate([0, 0, 1, 0, 1, 0, 1], [0, 0, 1, 0, 1, 0, 1])
        self.assertEqual(1, tnr)
        
        tnr = true_negative_rate([0, 0, 1, 0, 1, 0, 1], [0, 0, 1, 0, 1, 1, 1])
        self.assertEqual(3/4, tnr)
        
        tnr = true_negative_rate([0, 0, 1, 0, 1, 0, 1], [0, 0, 1, 1, 1, 0, 1])
        self.assertEqual(3/4, tnr)
        
        tnr = true_negative_rate([0, 0, 1, 0, 1, 0, 1], [0, 0, 1, 1, 1, 1, 1])
        self.assertEqual(2/4, tnr)
        
        tnr = true_negative_rate([0, 0, 1, 0, 1, 0, 1], [0, 1, 1, 1, 1, 1, 1])
        self.assertEqual(1/4, tnr)
        
        tnr = true_negative_rate([0, 0, 1, 0, 1, 0, 1], [1, 1, 1, 1, 1, 1, 1])
        self.assertEqual(0, tnr)
        
    def test_binary_confusion_matrix(self):
        tn, fp, fn, tp = binary_confusion_matrix([0, 0, 1, 1, 0, 0, 1, 1], [0, 0, 1, 1, 0, 0, 1, 1])
        self.assertListEqual([4, 0, 0, 4], [tn, fp, fn, tp])
        
        tn, fp, fn, tp = binary_confusion_matrix([0, 0, 1, 1, 0, 0, 1, 1], [1, 1, 0, 0, 1, 1, 0, 0])
        self.assertListEqual([0, 4, 4, 0], [tn, fp, fn, tp])
        
        tn, fp, fn, tp = binary_confusion_matrix([0, 0, 1, 1, 0, 0, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0])
        self.assertListEqual([4, 0, 4, 0], [tn, fp, fn, tp])
        
        tn, fp, fn, tp = binary_confusion_matrix([0, 0, 1, 1, 0, 0, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1])
        self.assertListEqual([0, 4, 0, 4], [tn, fp, fn, tp])
        
        tn, fp, fn, tp = binary_confusion_matrix([0, 0, 1, 1, 0, 0, 1, 1], [0, 0, 1, 1, 1, 1, 0, 0])
        self.assertListEqual([2, 2, 2, 2], [tn, fp, fn, tp])
