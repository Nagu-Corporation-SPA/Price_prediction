
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.metrics import root_mean_squared_error


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


