import os
from google.cloud import bigquery
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, root_mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
import pmdarima
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings


# convierte df a promedio de precios mensuales
"""
Filtra el DataFrame por el producto 'UB Atlantic TRIM-D 3-4 Lb FOB Miami' y devuelve  
el promedio mensual de precios, crea  'year_month' lo convierte a date y luego a index
"""
def filtrar_ub_mensual(df: pd.DataFrame) -> pd.Series:
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["priceName"] == "UB Atlantic TRIM-D 3-4 Lb FOB Miami"].copy()
    df["year_month"] = df["date"].dt.to_period("M").astype(str)

    df = df.groupby("year_month")["price"].mean().reset_index()
    df["year_month"] = pd.to_datetime(df["year_month"])
    df = df.set_index("year_month")

    return df["price"]

# convierte df a promedio de precios semanales
def filtrar_ub_semanal(df: pd.DataFrame) -> pd.Series:
    """
    Filtra el DataFrame por el producto 'UB Atlantic TRIM-D 3-4 Lb FOB Miami' y devuelve  
    el promedio semanal de precios, agrupado por año y semana.
    """
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["priceName"] == "UB Atlantic TRIM-D 3-4 Lb FOB Miami"].copy()

    df["iso_year"] = df["date"].dt.isocalendar().year
    df["iso_week"] = df["date"].dt.isocalendar().week
    semanal = df.groupby(["iso_year", "iso_week"])["price"].mean()

    # Convertir a serie con índice de tipo fecha usando el lunes de cada semana ISO
    semanal.index = [pd.to_datetime(f"{year}-W{int(week)}-1", format="%G-W%V-%u") for year, week in semanal.index]
    semanal = semanal.sort_index()

    return semanal





# Filtra df por trimestre lo que devuelve df_1, df_2, df_3, df_4
"""
    Filtra un DataFrame para quedarte solo con un trimestre específico del año.
    
    Args:
        df: DataFrame con índice de tipo datetime.
        trimestre: del 1 a 4
    
    Returns:
        DataFrame filtrado por trimestre
"""
def filtrar_trimestre(df: pd.DataFrame, trimestre: int = 1) -> pd.DataFrame:
    # Diccionario con los meses por trimestre
    trimestres = {
        1: [1, 2, 3],
        2: [4, 5, 6],
        3: [7, 8, 9],
        4: [10, 11, 12]
    }

    if trimestre not in trimestres:
        raise ValueError("Trimestre debe ser un número entre 1 y 4")

    meses = trimestres[trimestre]
    df_trim = df[df.index.month.isin(meses)].sort_index()

    return df_trim


# Partir en train test
"""
    Separa una serie temporal en conjuntos de entrenamiento y prueba.

    Args:
        df_trimestre (pd.Series): Serie temporal de un trimestre específico.
        test_size (int): Cantidad de observaciones para el conjunto de prueba.

    Returns:
        tuple: (train, test) como pandas.Series
"""
def partir_train_test(df_trimestre: pd.Series, test_size: int = 3): # toma los últimos 3 meses como test

    train = df_trimestre.iloc[:-test_size]
    test = df_trimestre.iloc[-test_size:]
    return train, test






