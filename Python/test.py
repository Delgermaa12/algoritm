import unittest
from scipy.spatial import cKDTree

def nearest_node_mock(point, nodes, tree):
    dist, idx = tree.query([point[0], point[1]], k=1)
    return nodes[int(idx)]

class TestNearestNode(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.nodes = [(0,0), (0,1), (1,0), (1,1)]
        cls.tree = cKDTree(cls.nodes)

    def test_origin(self):
        point = (0.1, 0.1)
        nearest = nearest_node_mock(point, self.nodes, self.tree)
        self.assertEqual(nearest, (0,0))

    def test_corner(self):
        point = (0.9, 1.1)
        nearest = nearest_node_mock(point, self.nodes, self.tree)
        self.assertEqual(nearest, (1,1))

    def test_edge(self):
        point = (0.5, 0.2)
        nearest = nearest_node_mock(point, self.nodes, self.tree)
        self.assertIn(nearest, [(0,0),(1,0)])

if __name__ == "__main__":
    unittest.main()
