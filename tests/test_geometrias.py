"""Invariantes físicas y dimensionales de las geometrías."""

from __future__ import annotations

import unittest

from src.comparacion_tc import tabla_vecinos
from src.ising import crear_ising, dimension_red, tc_teorica


class TestGeometriasIsing(unittest.TestCase):
    def test_energias_ferromagneticas_por_sitio(self) -> None:
        casos = (
            ("cuadrada", 2, 1, 4, -2.0),
            ("triangular", 2, 1, 6, -3.0),
            ("cubica", 3, 1, 6, -3.0),
            ("bcc", 3, 2, 8, -4.0),
            ("fcc", 3, 4, 12, -6.0),
            ("fe_mp13", 3, 2, 8, -4.0),
            ("ni_mp23", 3, 4, 12, -6.0),
            ("co_mp102", 3, 4, 12, -6.0),
        )
        for geometria, dimension, sitios, vecinos, energia in casos:
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
                forma = (4,) * dimension
                if sitios > 1:
                    forma = (*forma, sitios)
                self.assertEqual(modelo.spins.shape, forma)
                self.assertEqual(modelo.Nspin, sitios * 4**dimension)
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
        self.assertAlmostEqual(tc_teorica("bcc"), 6.3558)
        self.assertAlmostEqual(tc_teorica("fcc"), 9.794)
        self.assertAlmostEqual(tc_teorica("fe_mp13"), 6.3558)
        self.assertAlmostEqual(tc_teorica("ni_mp23"), 9.794)
        self.assertAlmostEqual(tc_teorica("co_mp102"), 9.794)

    def test_tabla_vecinos_cubica_para_comparacion_tc(self) -> None:
        vecinos = tabla_vecinos("cubica", 4)
        self.assertEqual(dimension_red("cubica"), 3)
        self.assertEqual(vecinos.shape, (4**3, 6))

    def test_topologias_cubicas_centradas(self) -> None:
        self.assertEqual(tabla_vecinos("bcc", 4).shape, (2 * 4**3, 8))
        self.assertEqual(tabla_vecinos("fcc", 4).shape, (4 * 4**3, 12))
        self.assertEqual(tabla_vecinos("fe_mp13", 4).shape, (2 * 4**3, 8))
        self.assertEqual(tabla_vecinos("ni_mp23", 4).shape, (4 * 4**3, 12))
        self.assertEqual(tabla_vecinos("co_mp102", 4).shape, (4 * 4**3, 12))

        bcc = crear_ising(
            "bcc",
            L=4,
            J=-1.0,
            h=0.0,
            T=1.0,
            inicializacion="antiferromagnetico",
            seed=3,
        )
        self.assertEqual(bcc.Nspin, 2 * 4**3)
        self.assertAlmostEqual(bcc.E / bcc.Nspin, -4.0)
        self.assertAlmostEqual(bcc.m / bcc.Nspin, 0.0)


if __name__ == "__main__":
    unittest.main()
