import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pyswarm import pso
from sklearn.svm import SVR

X = pd.read_csv("data/features.csv").values
y = pd.read_csv("data/target.csv").values
X = np.nan_to_num(X, nan=np.nanmean(X))

if y.ndim > 1 and y.shape[1] > 1:
    y = y[:, -1]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


class MLSSVR:
    def __init__(self, C, epsilon, gamma, layers):
        self.layers = layers
        self.epsilon = epsilon
        self.gamma = gamma
        self.C = C
        self.models = [SVR(C=C, epsilon=epsilon, gamma=gamma) for _ in range(layers)]

    def fit(self, X, y):
        output = X
        self.train_losses = []
        for model in self.models:
            model.fit(output, y)
            y_pred = model.predict(output)
            self.train_losses.append(mean_squared_error(y, y_pred))
            output = y_pred.reshape(-1, 1)

    def predict(self, X):
        output = X
        for model in self.models:
            output = model.predict(output).reshape(-1, 1)
        return output.flatten()

    def save(self, filename):
        import joblib
        joblib.dump(self, filename)


def pso_objective(params):
    C, epsilon, gamma = params[:3]
    layers = int(params[3])
    model = MLSSVR(C=C, epsilon=epsilon, gamma=gamma, layers=layers)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_train)
    mse = mean_squared_error(y_train, y_pred)

    return mse


lb = [0.1, 0.001, 0.001, 1]
ub = [10, 1, 1, 10]

best_params, _ = pso(pso_objective, lb, ub, swarmsize=10, maxiter=30)

C, epsilon, gamma, layers = best_params
layers = int(layers)

optimized_model = MLSSVR(C, epsilon, gamma, layers)
optimized_model.fit(X_train, y_train)

y_pred = optimized_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Test set performance: MSE={mse}, MAE={mae}, R2={r2}")

optimized_model.save("SlurryDeformationPrediction_MLSSVR.pkl")


plt.scatter(y_test, y_pred, label='Predictions')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Predicted vs Actual Values')
plt.legend()
plt.savefig('Predicted_vs_Actual_Values.png')
plt.show()


# Training Loss
plt.plot(optimized_model.train_losses, label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()
plt.savefig('Training_Loss.png')
plt.show()