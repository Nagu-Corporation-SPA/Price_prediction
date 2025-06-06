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

# Descargar Global prices as df
'''
Ejemplo de uso
query = SELECT * FROM TABLENAME
dfTable = downloadTable(query,nombreBonitoQuery)
forceDownload es util si es que los datos de bigquery fueron actualizados
'''
def downloadTable(query,queryName, forceDownload=False):
    folderPath = "bigqueryDatabases"
    if not os.path.exists(folderPath):
        os.makedirs(folderPath)  # Creates the folder (and any intermediate directories)
        print(f"Folder '{folderPath}' created.")
    else:
        print(f"Folder '{folderPath}' already exists.")

    fileName = queryName +".csv"
    filePath = os.path.join(folderPath, fileName)

    # If file exists and not forcing download, read from CSV
    if os.path.exists(filePath) and not forceDownload:
        print(f"Reading {filePath} from local CSV.")
        return pd.read_csv(filePath)

    # Otherwise, query BigQuery
    print(f"Querying from BigQuery...")
    clientBq = bigquery.Client()
    queryText = query
    queryJob = clientBq.query_and_wait(queryText)
    dfTable = queryJob.to_dataframe()

    # Ensure the directory exists
    os.makedirs(folderPath, exist_ok=True)

    # Save the result as a CSV
    dfTable.to_csv(filePath, index=False)
    print(f"Saved table to {filePath}")
    
    return dfTable

# convierte df a promedio de precios mensuales
"""
Filtra el DataFrame por el producto 'UB Atlantic TRIM-D 3-4 Lb FOB Miami' y devuelve  
el promedio mensual de precios, crea  'year_month' lo convierte a date y luego a index
"""
def filtrar_ub(df: pd.DataFrame) -> pd.Series:
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["priceName"] == "UB Atlantic TRIM-D 3-4 Lb FOB Miami"].copy()
    df["year_month"] = df["date"].dt.to_period("M").astype(str)

    # Agrupar por mes y calcular promedio
    df = df.groupby("year_month")["price"].mean().reset_index()

    # Convertir a datetime y establecer como índice
    df["year_month"] = pd.to_datetime(df["year_month"])
    df = df.set_index("year_month")

    return df["price"]


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


# Mehores hiperparámetros para ARIMA
def grid_search_arima_aic(series, p_range, d_range, q_range):
    best_aic = np.inf
    best_order = None
    best_model = None

    for p in p_range:
        for d in d_range:
            for q in q_range:
                try:
                    model = ARIMA(series, order=(p, d, q))
                    fitted = model.fit()
                    if fitted.aic < best_aic:
                        best_aic = fitted.aic
                        best_order = (p, d, q)
                        best_model = fitted
                except:
                    continue

    return best_order, best_aic

#Mejores hiperparámetros para SARIMA
def grid_search_sarima(series, p_range, d_range, q_range, P_range, D_range, Q_range, s, verbose=False):
    
    best_aic = np.inf
    best_order = None
    best_seasonal_order = None
    best_model = None
    
    warnings.filterwarnings("ignore")
    
    for p in p_range:
        for d in d_range:
            for q in q_range:
                for P in P_range:
                    for D in D_range:
                        for Q in Q_range:
                            order = (p, d, q)
                            seasonal_order = (P, D, Q, s)
                            try:
                                model = SARIMAX(series,
                                                order=order,
                                                seasonal_order=seasonal_order,
                                                enforce_stationarity=False,
                                                enforce_invertibility=False)
                                results = model.fit(disp=False)
                                
                                if results.aic < best_aic:
                                    best_aic = results.aic
                                    best_order = order
                                    best_seasonal_order = seasonal_order
                                    best_model = results
                                if verbose:
                                    print(f'Tested SARIMA{order}x{seasonal_order} - AIC={results.aic:.2f}')
                            except:
                                continue

    return {
        "order": best_order,
        "seasonal_order": best_seasonal_order,
        "aic": best_aic,
        "model": best_model
    }


# Función para obtener las metricas de error
def errorMetrics(real:np.array,pred:np.array):
    mse = mean_squared_error(real,pred)
    mae = mean_absolute_error(real,pred)
    rmse = root_mean_squared_error(real,pred)
    mape = mean_absolute_percentage_error(real,pred)
    return {"mse":mse,
            "mae":mae,
            "rmse":rmse,
            "mape":mape}
    
   

## Función para enventar la serie de tiempo
#def slidingWindows(serie:np.array,windowSize:int):
#    
#
#    return 0



