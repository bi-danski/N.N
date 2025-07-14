import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import custom_object_scope
import joblib


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
    X_test = np.sin(2 * X_test)
    return X_test


with custom_object_scope({'RBFLayer': RBFLayer}):
    model = load_model("SlurryDeformationPrediction_RBF.h5")

scaler = joblib.load("scaler.pkl")

X_test = pd.read_csv("data/features.csv").values

X_test = np.nan_to_num(X_test, nan=np.nanmean(X_test))
X_test_scaled = scaler.transform(X_test)
X_test_scaled = randomize(X_test_scaled)
y_pred = model.predict(X_test_scaled)

print(y_pred)