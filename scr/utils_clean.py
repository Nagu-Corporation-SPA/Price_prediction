
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

    return df["price"]/2.204


# convierte df a promedio de precios semanales
def filtrar_ub_semanal(df: pd.DataFrame) -> pd.Series:
    """
    Filtra el DataFrame para el producto
    'UB Atlantic TRIM-D 3-4 Lb FOB Miami' y devuelve la serie
    de precios promediados por semana ISO (lunes).

    • El índice resultante es 'W-MON'.
    • Las semanas sin datos quedan como NaN.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    df = df[df["priceName"] == "UB Atlantic TRIM-D 3-4 Lb FOB Miami"].copy()

    # --- Agrupación por semana ISO (lunes) ----
    semanal = (
        df.set_index("date")
          .groupby(pd.Grouper(freq="W-MON"))["price"]
          .mean()
          .asfreq("W-MON")          # fuerza semanas ausentes → NaN
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


# convierte df a promedio de precios semanales y crea features de retardos y medias móviles para el XGBoost
def filtrar_y_crear_features_semanal(
    df: pd.DataFrame,
    weeks_lags: list[int]    = [1, 2, 3, 4],   # retardos en semanas
    weeks_windows: list[int] = [4, 12],       # medias móviles en semanas
    obs_per_week: int        = 2,             # 2 observaciones por semana
) -> pd.DataFrame:
    """
    Filtra por 'UB Atlantic TRIM-D 3-4 Lb FOB Miami', agrupa semanalmente (promedio
    de las 2 observaciones), y crea features de retardos y medias móviles donde
    1 semana = `obs_per_week` datos.
    """
    # 1) Filtrado y agrupación semanal
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['priceName']=='UB Atlantic TRIM-D 3-4 Lb FOB Miami']
    df = df.set_index('date').sort_index()
    
    semanal = (
        df['price']
        .resample('W-MON')    # ISO‐semana, índice en lunes
        .mean()               # promedio de las ~2 obs que caen en la semana
        .dropna()
        .rename('price')
    )
    
    # 2) DataFrame para features
    df_feat = semanal.to_frame()
    
    # opcional: features de calendario
    df_feat['year']       = df_feat.index.year
    df_feat['weekofyear'] = df_feat.index.isocalendar().week.astype(int)
    
    # 3) Retardos en semanas (con shift = semanas * obs_per_week)
    for w in weeks_lags:
        df_feat[f'lag_{w}wk'] = df_feat['price'].shift(w * obs_per_week)
    
    # 4) Medias móviles en semanas (ventana = semanas * obs_per_week)
    for w in weeks_windows:
        df_feat[f'ma_{w}wk'] = df_feat['price'].rolling(window=w * obs_per_week).mean()
    
    # 5) Limpiar NaNs
    return df_feat.dropna()




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






