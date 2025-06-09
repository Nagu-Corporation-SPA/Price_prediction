import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from utils_clean import filtrar_ub_mensual
from utils_metrics import errorMetrics
from utils_model import buscar_hiperparametros_arima_sarima

# --------------- ARIMA MENSUAL -----------------#
def entrenar_arima_mensual(path_csv: str, fecha_inicio_pred: str = '2025-01-01', fecha_fin_pred: str = '2025-12-01'):
    """
    Entrena un modelo ARIMA mensual, genera predicciones en rolling forecast y evalúa por cuartil.
    
    Parámetros:
    -----------
    path_csv : str
        Ruta al CSV con la serie temporal mensual
    fecha_inicio_pred : str
        Fecha inicial de predicción (inclusive)
    fecha_fin_pred : str
        Fecha final de predicción (inclusive)
    
    Retorna:
    --------
    pred_df : pd.DataFrame
        DataFrame con columnas 'real', 'predicho', 'cuartil_str' y fechas como índice
    """
    # leer y filtrar data a mensual
    df = pd.read_csv(path_csv, index_col=0, parse_dates=True)
    df = filtrar_ub_mensual(df)

    #  Separar train y test (hasta dic 2024)
    df = df.sort_index()
    train = df.loc[:'2024-12-01']

    #  mejores hiperparámetros
    modelo, order, seasonal_order, resumen = buscar_hiperparametros_arima_sarima(train, m=12, seasonal=True)
    print("Resumen búsqueda ARIMA:")
    print(resumen)
    print("Mejores (p, d, q):", order)
    print("Mejores (P, D, Q, m):", seasonal_order)

    #  Rolling forecast
    serie_rolling = train.copy()
    predicciones = []
    fechas_pred = pd.date_range(fecha_inicio_pred, fecha_fin_pred, freq='MS')

    for fecha in fechas_pred:
        modelo = ARIMA(serie_rolling, order=order)
        fitted = modelo.fit()
        pred = fitted.forecast(steps=1)[0]
        predicciones.append(pred)

        if fecha in df.index:
            valor_a_agregar = df.loc[fecha]
        else:
            valor_a_agregar = pred

        serie_rolling = pd.concat([serie_rolling, pd.Series([valor_a_agregar], index=[fecha])])

    # resultados
    reales_disponibles = [df[fecha] if fecha in df.index else np.nan for fecha in fechas_pred]
    pred_df = pd.DataFrame({'real': reales_disponibles, 'predicho': predicciones}, index=fechas_pred)

    # Evaluación por cuartil
    pred_df['año'] = pred_df.index.year
    pred_df['cuartil'] = pred_df.index.quarter
    pred_df['cuartil_str'] = pred_df['año'].astype(str) + '-Q' + pred_df['cuartil'].astype(str)

    print("\n📊 Evaluación por cuartil:")
    for cuartil in pred_df['cuartil_str'].unique():
        datos = pred_df[pred_df['cuartil_str'] == cuartil]
        mask = ~datos['real'].isna()
        if mask.sum() > 0:
            metrics = errorMetrics(datos.loc[mask, 'real'], datos.loc[mask, 'predicho'])
            print(f"{cuartil}: {metrics}")
        else:
            print(f"{cuartil}: Sin datos reales para evaluar.")

    return pred_df



# -----------------------------------------------#


# --------------- ARIMA SEMANAL -----------------#

# -----------------------------------------------#