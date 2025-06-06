from statsmodels.tsa.stattools import pacf
from statsmodels.graphics.tsaplots import plot_pacf
import numpy as np
import matplotlib.pyplot as plt
import pmdarima


data = np.random.randn(50)
fig, ax = plt.subplots()
plot_pacf(data, ax=ax, lags=20, method="ywm")
plt.show()

