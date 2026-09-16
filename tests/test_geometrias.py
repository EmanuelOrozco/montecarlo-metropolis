"""Invariantes físicas y dimensionales de las geometrías."""

from __future__ import annotations

import unittest

from src.ising import crear_ising, tc_teorica


class TestGeometriasIsing(unittest.TestCase):
    def test_energias_ferromagneticas_por_sitio(self) -> None:
        casos = (
            ("cuadrada", 2, 4, -2.0),
            ("triangular", 2, 6, -3.0),
            ("cubica", 3, 6, -3.0),
        )
        for geometria, dimension, vecinos, energia in casos:
            with self.subTest(geometria=geometria):
                modelo = crear_ising(
                    geometria,
                    L=4,
                    J=1.0,
                    h=0.0,
                    T=1.0,
                    inicializacion="ferromagnetico",
                    seed=1,
                )
                self.assertEqual(modelo.spins.shape, (4,) * dimension)
                self.assertEqual(modelo.Nspin, 4**dimension)
                self.assertEqual(modelo.z_vecinos, vecinos)
                self.assertAlmostEqual(modelo.E / modelo.Nspin, energia)
                self.assertAlmostEqual(modelo.m / modelo.Nspin, 1.0)

    def test_cubica_antiferromagnetica(self) -> None:
        modelo = crear_ising(
            "cubica",
            L=4,
            J=-1.0,
            h=0.0,
            T=1.0,
            inicializacion="antiferromagnetico",
            seed=2,
        )
        self.assertAlmostEqual(modelo.E / modelo.Nspin, -3.0)
        self.assertAlmostEqual(modelo.m / modelo.Nspin, 0.0)
        self.assertEqual(modelo._suma_vecinos(0, 0, 0), -6)

    def test_referencia_critica_cubica(self) -> None:
        self.assertAlmostEqual(tc_teorica("cubica"), 4.511524)


if __name__ == "__main__":
    unittest.main()
