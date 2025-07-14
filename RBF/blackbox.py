import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import load_model
from joblib import load
import matplotlib.pyplot as plt
from tensorflow.keras.utils import custom_object_scope


class RBFLayer(tf.keras.layers.Layer):
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


def randomize(X_test):
    # X_test = np.sin(2 * X_test)
    X_test = np.cos(X_test) ** 2
    return X_test


with custom_object_scope({'RBFLayer': RBFLayer}):
    model = load_model("SlurryDeformationPrediction_RBF.h5")
    scaler = load("scaler.pkl")


def black_box_testing(test_data, expected_output):
    test_data_scaled = scaler.transform(test_data)

    predictions = model.predict(test_data_scaled)

    mse = mean_squared_error(expected_output, predictions)
    mae = mean_absolute_error(expected_output, predictions)
    r2 = r2_score(expected_output, predictions)

    test_results = {
        "mse": mse,
        "mae": mae,
        "r2": r2,
        "predictions": predictions,
        "actual_values": expected_output
    }

    return test_results


test_data = pd.read_csv("data/features.csv").values
y = pd.read_csv("data/target.csv").values
if y.ndim > 1 and y.shape[1] > 1:
    y = y[:, 0]

expected_output = y

test_results = black_box_testing(test_data, expected_output)
print(f"Black Box Testing - Performance Metrics on Test Data:")
print(f"Test set performance:\nMSE={test_results['mse']}\nMAE={test_results['mae']}\nR2={test_results['r2']}")


plt.scatter(test_results['actual_values'], test_results['predictions'], marker="*")
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Predicted vs Actual Values (Black Box Testing)')
plt.legend()
plt.show()
