import unittest
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from SDP_PSO_MLSSVR import MLSSVR, pso_objective


class TestMLSSVR(unittest.TestCase):

    def setUp(self):
        self.X = np.array([[1, 2], [3, 4], [5, 6]])
        self.y = np.array([1, 2, 3])

        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(self.X)

    def test_init(self):
        C = 1.0
        epsilon = 0.1
        gamma = 0.5
        layers = 3
        model = MLSSVR(C=C, epsilon=epsilon, gamma=gamma, layers=layers)
        self.assertEqual(model.C, C)
        self.assertEqual(model.epsilon, epsilon)
        self.assertEqual(model.gamma, gamma)
        self.assertEqual(model.layers, layers)
        self.assertEqual(len(model.models), layers)
        self.assertTrue(all(isinstance(m, SVR) for m in model.models))
        print("[+] Test Complete")

    def test_fit(self):
        C = 1.0
        epsilon = 0.1
        gamma = 0.5
        layers = 2
        model = MLSSVR(C=C, epsilon=epsilon, gamma=gamma, layers=layers)
        model.fit(self.X_scaled, self.y)
        self.assertTrue(hasattr(model, 'train_losses'))
        self.assertEqual(len(model.train_losses), layers)
        print("[+] Test Complete")

    def test_predict(self):
        C = 1.0
        epsilon = 0.1
        gamma = 0.5
        layers = 2
        model = MLSSVR(C=C, epsilon=epsilon, gamma=gamma, layers=layers)
        model.fit(self.X_scaled, self.y)
        y_pred = model.predict(self.X_scaled)
        self.assertEqual(y_pred.shape, (len(self.X),))
        print("[+] Test Complete")

    def test_pso_objective(self):
        params = [1.0, 0.1, 0.5, 2]
        mse = pso_objective(params)
        self.assertIsInstance(mse, float)
        print("[+] Test Complete")

if __name__ == '__main__':
    unittest.main()