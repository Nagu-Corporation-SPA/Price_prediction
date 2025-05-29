from utils import downloadTable

def main():
    query = "SELECT * FROM `desarrollo-444913.globalPrices.prices`"
    nombre = "prices"
    df = downloadTable(query, nombre, forceDownload=False)
    
    print("Primeras filas del dataframe:")
    print(df.head())

if __name__ == "__main__":
    main()

