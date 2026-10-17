"""Pruebas del núcleo matemático (unittest, sin dependencias extra): python -m unittest discover -s tests"""
import sys, os, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from core.trajectory import Params
from core.constraints import Limits
from core.analysis import analyze
from core import field, local
from core.errors import DegenerateGeometryError, InvalidParameterError


class TestCore(unittest.TestCase):
    def test_helix_closed_forms(self):
        an = analyze(Params(2, 0, 0.5), Limits())
        self.assertAlmostEqual(an.k.curvature[0], 2 / (4 + 0.25), places=9)
        self.assertAlmostEqual(an.arc, np.sqrt(4.25) * 4 * np.pi, places=6)

    def test_original_violations(self):
        an = analyze(Params(), Limits())
        self.assertEqual({v.key for v in an.violations}, {"speed", "accel", "temp"})

    def test_proposals_comply(self):
        for abc in [(2.2, 0, 0.1), (2.1, 0.02, 0.1)]:
            self.assertTrue(analyze(Params(*abc), Limits()).ok, abc)

    def test_frenet_orthonormal(self):
        k = analyze(Params(), Limits()).k
        for U, V in [(k.Tn, k.N), (k.Tn, k.B), (k.N, k.B)]:
            self.assertTrue(np.allclose(np.sum(U * V, axis=1), 0, atol=1e-9))
        self.assertTrue(np.allclose(np.linalg.norm(k.N, axis=1), 1))

    def test_gradient_numeric(self):
        p = np.array([0.7, -1.2, 0.4]); h = 1e-6
        num = [(field.T_of(p + h * e) - field.T_of(p - h * e)) / (2 * h) for e in np.eye(3)]
        self.assertTrue(np.allclose(field.gradient(p), num, atol=1e-5))

    def test_quadratic_is_exact(self):
        p0 = np.array([1.0, 2.0, 0.5])
        for r in local.compare_along_direction(p0, [1, 1, 1], [0.1, 1.0]):
            self.assertAlmostEqual(r["error_cuadratica"], 0, places=9)

    def test_errors(self):
        with self.assertRaises(DegenerateGeometryError): analyze(Params(0, 0, 0), Limits())
        with self.assertRaises(DegenerateGeometryError): analyze(Params(0, 0, 1), Limits())
        with self.assertRaises(InvalidParameterError): analyze(Params(float("nan"), 0, 1), Limits())
        with self.assertRaises(InvalidParameterError): analyze(Params(), Limits(speed_max=0))

    def test_time_violation(self):
        an = analyze(Params(2.2, 0, 0.1, 5 * np.pi), Limits())
        self.assertIn("time", {v.key for v in an.violations})


if __name__ == "__main__":
    unittest.main()
