"""Función de ganancia del negocio: +25% del monto por legítima aprobada y -100% por fraude aprobado."""

import numpy as np
import pytest

from fraude_shipping.ganancia import UMBRALES, calcular_ganancia, curva_ganancia, metricas_decision, umbral_optimo

FRAUDE = np.array([0, 0, 1, 1])
MONTO = np.array([100.0, 200.0, 50.0, 40.0])


def test_ganancia_con_valores_calculados_a_mano():
    """Aprobar una legítima de 100 y un fraude de 50 da 0,25 * 100 - 50."""
    aprobada = np.array([True, False, True, False])
    assert calcular_ganancia(FRAUDE, MONTO, aprobada) == pytest.approx(-25.0)


def test_rechazar_todo_da_ganancia_cero():
    """Una transacción rechazada no suma ni resta."""
    assert calcular_ganancia(FRAUDE, MONTO, np.zeros(4, dtype=bool)) == 0


def test_ganancia_maxima_aprueba_solo_legitimas():
    """La ganancia máxima es el 25% de la suma de los montos legítimos."""
    assert calcular_ganancia(FRAUDE, MONTO, FRAUDE == 0) == pytest.approx(75.0)


def test_curva_ganancia_tiene_un_valor_por_umbral():
    """La curva evalúa la ganancia en cada umbral pedido."""
    probabilidad = np.array([0.1, 0.2, 0.8, 0.9])
    curva = curva_ganancia(FRAUDE, MONTO, probabilidad, [0.15, 0.5, 1.0])
    assert list(curva['umbral']) == [0.15, 0.5, 1.0]
    assert list(curva['ganancia']) == pytest.approx([25.0, 75.0, -15.0])


def test_umbral_optimo_separa_fraudes_de_legitimas():
    """Con probabilidades que separan perfecto, el umbral óptimo aprueba solo las legítimas."""
    probabilidad = np.array([0.1, 0.1, 0.9, 0.9])
    umbral = umbral_optimo(FRAUDE, MONTO, probabilidad)
    assert umbral in UMBRALES
    assert 0.1 < umbral <= 0.9
    assert calcular_ganancia(FRAUDE, MONTO, probabilidad < umbral) == pytest.approx(75.0)


def test_metricas_decision():
    """Ganancia relativa a la máxima, % de aprobadas y % de fraudes detectados."""
    aprobada = np.array([True, False, True, False])
    metricas = metricas_decision(FRAUDE, MONTO, aprobada)
    assert metricas['ganancia_pct_maxima'] == pytest.approx(-25.0 / 75.0 * 100)
    assert metricas['aprobadas_pct'] == pytest.approx(50.0)
    assert metricas['fraudes_detectados_pct'] == pytest.approx(50.0)
