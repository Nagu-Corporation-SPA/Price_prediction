from utils import downloadTable, errorMetrics

import numpy as np

real = np.array([100, 200, 300])
pred = np.array([110, 190, 310])

metrics = errorMetrics(real, pred)
print("Métricas de error:", metrics)

