
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss




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
def filtrar_ub_semanal_iso(df: pd.DataFrame) -> pd.Series:
    """
    Filtra el DataFrame para el producto
    'UB Atlantic TRIM-D 3-4 Lb FOB Miami' y devuelve la serie
    de precios promediados por semana ISO (año + semana).

    • El índice es un datetime correspondiente al lunes de la semana ISO.
    • Las semanas sin datos quedan como NaN.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    
    # Filtrar el producto deseado
    df = df[df["priceName"] == "UB Atlantic TRIM-D 3-4 Lb FOB Miami"].copy()
    
    # Extraer año y semana ISO
    iso = df["date"].dt.isocalendar()
    df["iso_year"] = iso["year"]
    df["iso_week"] = iso["week"]
    
    # Calcular el lunes correspondiente a cada semana ISO
    df["iso_monday"] = pd.to_datetime(df["iso_year"].astype(str) + "-" + df["iso_week"].astype(str) + "-1", format="%G-%V-%u")

    # Agrupar por lunes de semana ISO
    semanal = (
        df.groupby("iso_monday")["price"]
          .mean()
          .asfreq("W-MON")  # asegura presencia de semanas faltantes con NaN
          .sort_index()
    )
    
    return semanal


# imputar nulos para el arima semanal
def imputar_nulos_semanal(
        serie: pd.Series,
        metodo: str = "interpolate",        # 'interpolate' | 'ffill' | 'bfill'
        devolver_flags: bool = False
) -> pd.Series | tuple[pd.Series, pd.Series]:
    """
    Imputa los NaN de una serie semanal:

        'interpolate' → interpolación lineal
        'ffill'       → relleno hacia adelante
        'bfill'       → relleno hacia atrás

    Si devolver_flags=True, también devuelve una serie booleana
    que marca con True los puntos imputados.
    """
    s = serie.copy()
    flags = s.isna()             # marca posiciones vacías

    if metodo == "interpolate":
        s = s.interpolate()
    elif metodo == "ffill":
        s = s.ffill()
    elif metodo == "bfill":
        s = s.bfill()
    else:
        raise ValueError("metodo debe ser 'interpolate', 'ffill' o 'bfill'")

    if devolver_flags:
        return s, flags
    return s



# VOlver estacionaria la serie semanal
def estacionarizar_arima(serie: pd.Series):
    """
    Aplica diferencias simples hasta que la serie sea estacionaria a d.
    Devuelve la serie transformada y el valor óptimo de d.
    """
    for d in range(3):  #  d = 0, 1, 2
        serie_diff = serie.diff(d).dropna() if d > 0 else serie.copy()
        
        adf_p = adfuller(serie_diff)[1]
        kpss_p = kpss(serie_diff, nlags="auto")[1]
        
        if adf_p < 0.05 and kpss_p > 0.05:
            print(f"Serie estacionaria con d = {d}")
            return serie_diff, d
    
    raise ValueError("No se logró encontrar un valor de d que haga la serie estacionaria.")


import pandas as pd
# Añade features para el XGBoost
def crear_features_temporales(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea múltiples features temporales a partir de una serie semanal con columna 'price'.

    Parámetro:
    - df: DataFrame con índice datetime semanal y columna 'price'.

    Retorna:
    - df_feat: DataFrame con variables listas para modelar.
    """
    df = df.copy()
    dt_index = pd.DatetimeIndex(df.index)


    #  Features de calendario
    df["year"]        = dt_index.year
    df["month"]       = dt_index.month
    df["quarter"]     = dt_index.quarter
    df["weekofyear"]  = dt_index.isocalendar().week.astype(int)
    df["dayofyear"]   = dt_index.dayofyear
    
    #  Lags semanales
    for lag in [1, 2, 3, 4, 8, 12]:
        df[f"lag_{lag}wk"] = df["price"].shift(lag)

    #  Cambios absolutos y relativos
    df["diff_1wk"] = df["price"].diff(1)
    df["pct_change_1wk"] = df["price"].pct_change(1)

    #  Estadísticas móviles
    for w in [4, 8, 12]:
        df[f"ma_{w}wk"]     = df["price"].rolling(window=w).mean()
        df[f"std_{w}wk"]    = df["price"].rolling(window=w).std()
        df[f"min_{w}wk"]    = df["price"].rolling(window=w).min()
        df[f"max_{w}wk"]    = df["price"].rolling(window=w).max()
        df[f"median_{w}wk"] = df["price"].rolling(window=w).median()

    return df








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

# Funcion para preprocesar los datos para Prophet
def preprocess_data_prophet(df:pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame(df).reset_index()
    columns = list(df.columns)
    df = df.rename(columns={columns[0]: 'ds', columns[1]: 'y'})
    df['ds'] = pd.to_datetime(df['ds'])
    return df




