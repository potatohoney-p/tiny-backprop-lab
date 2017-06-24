import unittest
from core import Network
from features import gradient_error,dumps,loads
class FeatureTests(unittest.TestCase):
    def test_gradient_matches_finite_difference(self): self.assertLess(gradient_error(Network(),[0.3,0.7],1),1e-8)
    def test_gradient_check_preserves_weights(self):
        n=Network(); before=n.weights[:]; gradient_error(n,[0,1],0); self.assertEqual(n.weights,before)
    def test_roundtrip(self):
        n=Network(); self.assertEqual(loads(dumps(n)).predict([1,0]),n.predict([1,0]))
