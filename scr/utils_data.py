import os
import pandas as pd
from google.cloud import bigquery
from sklearn.model_selection import train_test_split
import requests
import json
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
