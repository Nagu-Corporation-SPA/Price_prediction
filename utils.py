import os
from google.cloud import bigquery
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error,root_mean_squared_error

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
el promedio mensual de precios.
    
Args:
    df

Returns:
    pd.df:con 'year_month' y 'price' mensual
"""

def filtrar_ub(df: pd.DataFrame) -> pd.DataFrame:

    df["date"] = pd.to_datetime(df["date"])
    df = df[df["priceName"] == "UB Atlantic TRIM-D 3-4 Lb FOB Miami"].copy()
    df["year_month"] = df["date"].dt.to_period("M").astype(str)
    # Agrupar y calcular promedio mensual
    serie_mensual = df.groupby("year_month")["price"].mean().reset_index()
    return serie_mensual



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


    



