"""Pruebas Heisenberg clásico y validación MP."""

from __future__ import annotations

import unittest

import numpy as np

from src.heisenberg.espines import vector_aleatorio_esfera
from src.heisenberg.base import crear_heisenberg
from src.materials.estructura import analizar_estructura_mp


class TestHeisenberg(unittest.TestCase):
    def test_vectores_esfera_sin_sesgo_norma(self) -> None:
        rng = np.random.default_rng(0)
        v = vector_aleatorio_esfera(rng, size=(5000,))
        normas = np.linalg.norm(v, axis=1)
        self.assertTrue(np.allclose(normas, 1.0, atol=1e-12))
        # Media ~0 en cada componente (uniforme en esfera)
        self.assertLess(abs(v[:, 0].mean()), 0.05)
        self.assertLess(abs(v[:, 1].mean()), 0.05)
        self.assertLess(abs(v[:, 2].mean()), 0.05)

    def test_energia_ferromagnetica_bcc_mp(self) -> None:
        m = crear_heisenberg(
            "fe_mp13", L=4, J=1.0, T=1.0, h=0.0, kB=1.0,
            inicializacion="ferromagnetico", seed=1,
        )
        # Cada enlace aporta -J; z=8 → E/N = -z/2 = -4
        self.assertAlmostEqual(m.E / m.Nspin, -4.0, places=6)
        self.assertAlmostEqual(m.magnetizacion_por_sitio(), 1.0, places=6)

    def test_validacion_mp_fe_ni_co(self) -> None:
        fe = analizar_estructura_mp("fe_mp13")
        self.assertTrue(fe.valida)
        self.assertEqual(fe.tipo_red, "BCC")
        ni = analizar_estructura_mp("ni_mp23")
        self.assertTrue(ni.valida)
        self.assertEqual(ni.tipo_red, "FCC")
        co = analizar_estructura_mp("co_mp102")
        self.assertTrue(co.valida)
        self.assertEqual(co.tipo_red, "FCC")


if __name__ == "__main__":
    unittest.main()
