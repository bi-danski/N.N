import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler


def randomize(X_test):
    X_test = np.sin(2 * X_test)
    return X_test


scaler = StandardScaler()
loaded_model = load_model("SDP_GA-BP.h5")

X_test = pd.read_csv("data/training.csv")
X_test = X_test.values
X_test = np.nan_to_num(X_test, nan=np.nanmean(X_test))

X_test_scaled = scaler.fit_transform(X_test)
X_test = randomize(X_test)
y_pred = loaded_model.predict(X_test)

print(y_pred)


