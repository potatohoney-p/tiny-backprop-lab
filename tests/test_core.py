import unittest
from core import Network,sigmoid
class CoreTests(unittest.TestCase):
    def test_extreme_sigmoid(self): self.assertEqual(sigmoid(-1000),0); self.assertEqual(sigmoid(1000),1)
    def test_seed(self): self.assertEqual(Network().weights,Network().weights)
    def test_xor_learning(self):
        n=Network(); rows=[[0,0],[0,1],[1,0],[1,1]]; y=[0,1,1,0]
        n.train(rows,y,epochs=6000)
        self.assertEqual([int(n.predict(r)>0.5) for r in rows],y)
    def test_dimension(self):
        with self.assertRaises(ValueError): Network().predict([1])
