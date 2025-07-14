import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib


model = joblib.load("SlurryDeformationPrediction_MLSSVR.pkl")
X_test = pd.read_csv("features.csv").values
y_test = pd.read_csv("target.csv").values


X_test = np.nan_to_num(X_test, nan=np.nanmean(X_test))
if y_test.ndim > 1 and y_test.shape[1] > 1:
    y_test = y_test[:, -1]

scaler = StandardScaler()
X_test_scaled = scaler.fit_transform(X_test)

y_pred = model.predict(X_test_scaled)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Test set performance: MSE={mse}, MAE={mae}, R2={r2}")

plt.scatter(y_test, y_pred, label='Predictions')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Predicted vs Actual Values')
plt.legend()
plt.show()