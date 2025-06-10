
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.metrics import root_mean_squared_error


# Función para obtener las metricas de error
def errorMetrics(real: np.array, pred: np.array):
    # Element-wise errors
    errors = real - pred
    abs_errors = np.abs(errors)
    squared_errors = errors**2
    percentage_errors = np.abs(errors / real) * 100

    # Means
    mse = np.mean(squared_errors)
    mae = np.mean(abs_errors)
    rmse = np.sqrt(mse)
    mape = np.mean(percentage_errors)

    # Standard deviations
    mse_std = np.std(squared_errors)
    mae_std = np.std(abs_errors)
    rmse_std = np.std(np.sqrt(squared_errors))  # std of per-sample RMSE (i.e., sqrt(errors^2))
    mape_std = np.std(percentage_errors)

    return {
        "mse": mse,
        "mse_std": mse_std,
        "mae": mae,
        "mae_std": mae_std,
        "rmse": rmse,
        "rmse_std": rmse_std,
        "mape": mape,
        "mape_std": mape_std
    }
   

## Función para enventar la serie de tiempo
#def slidingWindows(serie:np.array,windowSize:int):
#    
#
#    return 0


