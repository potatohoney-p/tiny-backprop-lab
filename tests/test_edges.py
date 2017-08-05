import unittest
from core import Network
from features import loads
class EdgeTests(unittest.TestCase):
    def test_bad_target_no_mutation(self):
        n=Network(); before=n.weights[:]
        with self.assertRaises(ValueError): n.train([[0,1]],[2])
        self.assertEqual(n.weights,before)
    def test_invalid_serialization(self):
        with self.assertRaises(ValueError): loads('{"inputs":2,"hidden":1,"weights":[]}')
    def test_nonfinite_input(self):
        with self.assertRaises(ValueError): Network().predict([float('inf'),0])
