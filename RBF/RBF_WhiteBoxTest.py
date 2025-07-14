import unittest
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from SDP_RBF import create_rbf_model, pso_objective, RBFLayer
from sklearn.preprocessing import StandardScaler


class TestRBFModel(unittest.TestCase):

    def setUp(self):
        self.input_shape = 10
        self.units = 16
        self.gamma = 0.5
        self.X_train = np.random.rand(100, self.input_shape)
        self.y_train = np.random.rand(100, 1)
        self.X_test = np.random.rand(20, self.input_shape)
        self.y_test = np.random.rand(20, 1)
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)

    def test_create_rbf_model(self):
        model = create_rbf_model(self.input_shape, self.units, self.gamma)
        self.assertIsInstance(model, Sequential)
        self.assertEqual(len(model.layers), 2)
        self.assertIsInstance(model.layers[1], Dense)
        print("[+] Test Complete")

    def test_rbf_layer(self):
        rbf_layer = RBFLayer(self.units, self.gamma)
        rbf_layer.build((None, self.input_shape))
        self.assertIsNotNone(rbf_layer.mu)
        self.assertEqual(rbf_layer.mu.shape, (self.units, self.input_shape))
        print("[+] Test Complete")

    def test_pso_objective(self):
        params = [self.units, self.gamma]
        mse = pso_objective(params)
        self.assertIsInstance(mse, float)
        print("[+] Test Complete")

    def test_model_compile_and_fit(self):
        model = create_rbf_model(self.input_shape, self.units, self.gamma)
        model.compile(optimizer=Adam(), loss=MeanSquaredError())
        model.fit(self.X_train_scaled, self.y_train, epochs=1, batch_size=8, verbose=0)
        self.assertIsNotNone(model.history)
        print("[+] Test Complete")

    def test_model_predict(self):
        model = create_rbf_model(self.input_shape, self.units, self.gamma)
        model.compile(optimizer=Adam(), loss=MeanSquaredError())
        model.fit(self.X_train_scaled, self.y_train, epochs=1, batch_size=8, verbose=0)
        y_pred = model.predict(self.X_test_scaled)
        self.assertEqual(y_pred.shape, self.y_test.shape)
        print("[+] Test Complete")

if __name__ == '__main__':
    unittest.main()