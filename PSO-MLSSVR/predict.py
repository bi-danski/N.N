import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
from mlssvr import MLSSVR

def randomize(X_test):
    x_test = np.sin(2 * X_test)
    return x_test


X_test = pd.read_csv("data/features.csv").values
X_test = np.nan_to_num(X_test, nan=np.nanmean(X_test))
scaler = StandardScaler()
X_test_scaled = scaler.fit_transform(X_test)

x_test = randomize(X_test)

loaded_model = joblib.load("SlurryDeformationPrediction_MLSSVR.pkl")
y_pred = loaded_model.predict(x_test)

print(y_pred)

