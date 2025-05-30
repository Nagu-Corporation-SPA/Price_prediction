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




