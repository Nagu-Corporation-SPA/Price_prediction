from pmdarima import auto_arima
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
import warnings
warnings.filterwarnings("ignore")

import optuna
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import itertools

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




def buscar_hiperparametros_optuna(X, y, n_trials=50, cv_splits=3, random_state=42):
    """
    Busca los mejores hiperparámetros para XGBRegressor usando Optuna + TimeSeriesSplit.
    
    Solo devuelve los mejores hiperparámetros. NO entrena el modelo final ni calcula errores.

    Parámetros:
    - X, y: datos de entrenamiento
    - n_trials: cantidad de combinaciones a probar
    - cv_splits: cantidad de splits temporales
    - random_state: semilla para reproducibilidad

    Retorna:
    - diccionario con los mejores hiperparámetros
    """

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma": trial.suggest_float("gamma", 0, 2),
            "reg_alpha": trial.suggest_float("reg_alpha", 0, 2),
            "reg_lambda": trial.suggest_float("reg_lambda", 0, 2)
        }

        model = XGBRegressor(**params, random_state=random_state, objective='reg:squarederror')

        tscv = TimeSeriesSplit(n_splits=cv_splits)
        rmse_list = []

        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            mse = mean_squared_error(y_val, preds)
            rmse = np.sqrt(mse)
            rmse_list.append(rmse)

        return np.mean(rmse_list)

    # Ejecutar la optimización
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=random_state))
    study.optimize(objective, n_trials=n_trials)

    return study.best_params



def gridSearchSarimax(trainingData,seasonality:int):
    p = d = q = range(0, 2)
    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = [(x[0], x[1], x[2], seasonality) for x in list(itertools.product(p, d, q))]
    resultAic = None
    pParameters = 0
    dParameters = 0
    qParameters = 0
    spParameters = 0
    sdParameters = 0
    sqParameters = 0
    bestModel = None

    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                mod = SARIMAX(trainingData,
                                        order=param,
                                        seasonal_order=param_seasonal,
                                        enforce_stationarity=False,
                                        enforce_invertibility=False)
                results = mod.fit()

                if bestModel == None or results.aic < resultAic:
                    resultAic = results.aic
                    pParameters = param[0]
                    dParameters = param[1]
                    qParameters = param[2]
                    spParameters = param_seasonal[0]
                    sdParameters = param_seasonal[1]
                    sqParameters = param_seasonal[2]
                    bestModel = results
            except:
                continue

    print("Best AIC: ", resultAic)
    print("Best pParameter: ", pParameters)
    print("Best dParameter: ", dParameters)
    print("Best qParameter: ", qParameters)
    print("Best spParameter: ", spParameters)
    print("Best sdParameter: ", sdParameters)
    print("Best sqParameter: ", sqParameters)
    bestModel.summary()

    return bestModel, resultAic, pParameters, dParameters, qParameters, spParameters, sdParameters, sqParameters,seasonality
