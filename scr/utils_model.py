from pmdarima import auto_arima
import numpy as np
import pandas as pd

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
import warnings
warnings.filterwarnings("ignore")


# MEJORES HIPERPARAMETROS ARIMA SARIMA MENSUAL
def buscar_hiperparametros_arima_sarima(series, 
                                 m=12, 
                                 seasonal=True, 
                                 max_p=5, max_d=2, max_q=5,
                                 max_P=2, max_D=1, max_Q=2,
                                 trace=True):
    """
    Busca los mejores hiperparámetros (p,d,q)(P,D,Q,m) para un modelo ARIMA/SARIMA
    usando auto_arima de pmdarima.

    Args:
        series: Serie temporal tipo pandas.Series
        m: Periodicidad de la estacionalidad (12 para meses con ciclo anual)
        seasonal: True si la serie tiene estacionalidad
        max_p, max_d, max_q: rangos máximos para los parámetros no estacionales
        max_P, max_D, max_Q: rangos máximos para los parámetros estacionales
        trace: True para mostrar el proceso

    Returns:
        modelo: modelo auto_arima ya ajustado
        order: tuple, mejores (p, d, q)
        seasonal_order: tuple, mejores (P, D, Q, m)
        summary: Resumen del modelo (string)
    """
    modelo = auto_arima(
        series,
        seasonal=seasonal,
        m=m,
        start_p=0, max_p=max_p,
        start_d=0, max_d=max_d,
        start_q=0, max_q=max_q,
        start_P=0, max_P=max_P,
        start_D=0, max_D=max_D,
        start_Q=0, max_Q=max_Q,
        trace=trace,
        error_action='ignore',
        suppress_warnings=True,
        stepwise=True
    )
    return modelo, modelo.order, modelo.seasonal_order, modelo.summary()




# MEJORES HIPERPARAMETROS ARIMA SEMANAL
def buscar_hiperparametros_arima_semanal(
    serie,
    max_p=5, max_d=2, max_q=5,
    criterio='aic',
    trace=True
):
    """
    Busca los mejores hiperparámetros (p,d,q) para un modelo ARIMA SEMANAL usando auto_arima.

    Parámetros:
    - serie: Serie temporal semanal (ya estacionaria o preprocesada)
    - max_p, max_d, max_q: rangos máximos para p, d, q
    - criterio: 'aic' o 'bic'
    - trace: si se desea mostrar el progreso

    Retorna:
    - modelo entrenado
    - mejores hiperparámetros (p,d,q)
    - resumen del modelo
    """
    modelo = auto_arima(
        serie,
        start_p=0, max_p=max_p,
        start_d=0, max_d=max_d,
        start_q=0, max_q=max_q,
        seasonal=False,         # ← SOLO ARIMA
        information_criterion=criterio,
        trace=trace,
        error_action='ignore',
        suppress_warnings=True,
        stepwise=True
    )
    return modelo, modelo.order, modelo.summary()



