import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pyswarm import pso
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
import joblib
import matplotlib.pyplot as plt


X = pd.read_csv("data/features.csv").values
y = pd.read_csv("data/target.csv").values

if y.ndim > 1 and y.shape[1] > 1:
    y = y[:, -1]

X = np.nan_to_num(X, nan=np.nanmean(X))

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


class RBFLayer(layers.Layer):
    def __init__(self, units, gamma):
        super(RBFLayer, self).__init__()
        self.units = units
        self.gamma = tf.constant(gamma, dtype=tf.float32)

    def build(self, input_shape):
        self.mu = self.add_weight(name='mu',
                                  shape=(self.units, input_shape[-1]),
                                  initializer='uniform',
                                  trainable=True)

    def call(self, inputs):

        diff = tf.expand_dims(inputs, 1) - tf.expand_dims(self.mu, 0)
        l2 = tf.reduce_sum(tf.square(diff), axis=-1)
        res = tf.exp(-1 * self.gamma * l2)
        return res


def create_rbf_model(input_shape, units, gamma):
    model = keras.Sequential()
    model.add(layers.InputLayer(input_shape=(input_shape,)))
    model.add(RBFLayer(units, gamma))
    model.add(layers.Dense(1))
    return model


def pso_objective(params):
    units, gamma = int(params[0]), params[1]
    model = create_rbf_model(X_train.shape[1], units, gamma)

    model.compile(optimizer='adam', loss='mse')

    X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    model.fit(X_train_split, y_train_split, epochs=100, batch_size=32, verbose=0)

    y_pred = model.predict(X_val)
    mse = mean_squared_error(y_val, y_pred)

    return mse


lb = [10, 0.01]  # Lower bounds
ub = [100, 10]  # Upper bounds

best_params, _ = pso(pso_objective, lb, ub, swarmsize=50, maxiter=100)

units, gamma = int(best_params[0]), best_params[1]

optimized_model = create_rbf_model(X_train.shape[1], units, gamma)
optimized_model.compile(optimizer='adam', loss='mse')
history = optimized_model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)


y_pred = optimized_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Test set performance: MSE={mse}, MAE={mae}, R2={r2}")

optimized_model.save("SDP_RBF.h5")
joblib.dump(scaler, "scaler.pkl")

# Learning Curves
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Learning Curves')
plt.legend()
plt.savefig('Learning_Curves.png')
plt.show()

plt.scatter(y_test, y_pred)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Predicted vs Actual Values')
plt.legend()
plt.savefig('Predicted_vs_Actual_Values.png')
plt.show()
