import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import netnull  # noqa: E402


def random_net(directed: bool, n: int = 400, m: int = 800, seed: int = 1) -> netnull.Network:
    rng = np.random.default_rng(seed)
    # a skewed degree sequence, as in real interactomes
    w = rng.pareto(1.5, n) + 1
    a = rng.choice(n, 4 * m, p=w / w.sum()); b = rng.choice(n, 4 * m, p=w / w.sum())
    genes = [f"G{i}" for i in range(n)]
    net = netnull.from_pairs("toy", [genes[i] for i in a], [genes[i] for i in b], directed)
    net.edges = net.edges[:m]
    return net


def edge_set(net):
    return set(map(tuple, net.edges.tolist()))


class NullModelTests(unittest.TestCase):
    def check_simple(self, net):
        self.assertTrue((net.edges[:, 0] != net.edges[:, 1]).all(), "self-loop")
        self.assertEqual(len(edge_set(net)), len(net.edges), "duplicate edge")
        if not net.directed:
            self.assertTrue((net.edges[:, 0] < net.edges[:, 1]).all())

    def test_undirected_rewire_keeps_every_degree(self):
        net = random_net(False)
        r = netnull.null(net, "rewired", 3)
        self.check_simple(r)
        self.assertTrue((net.degree() == r.degree()).all())
        self.assertLess(len(edge_set(net) & edge_set(r)) / len(net.edges), 0.5)

    def test_directed_rewire_keeps_in_and_out_degree(self):
        net = random_net(True)
        r = netnull.null(net, "rewired", 3)
        self.check_simple(r)
        for col in (0, 1):
            self.assertTrue((np.bincount(net.edges[:, col], minlength=net.n)
                             == np.bincount(r.edges[:, col], minlength=net.n)).all())

    def test_labels_keep_degree_distribution_not_assignment(self):
        net = random_net(False)
        r = netnull.null(net, "labels", 3)
        self.check_simple(r)
        self.assertTrue((np.sort(net.degree()) == np.sort(r.degree())).all())
        self.assertFalse((net.degree() == r.degree()).all())

    def test_er_keeps_edge_count_only(self):
        for directed in (False, True):
            net = random_net(directed)
            r = netnull.null(net, "er", 3)
            self.check_simple(r)
            self.assertEqual(len(r.edges), len(net.edges))

    def test_empty_and_real(self):
        net = random_net(False)
        self.assertEqual(len(netnull.null(net, "empty").edges), 0)
        self.assertIs(netnull.null(net, "real"), net)

    def test_seeds_differ_and_repeat(self):
        net = random_net(False)
        a, b, c = (netnull.null(net, "rewired", s) for s in (0, 1, 0))
        self.assertNotEqual(edge_set(a), edge_set(b))
        self.assertEqual(edge_set(a), edge_set(c))


if __name__ == "__main__":
    unittest.main()
