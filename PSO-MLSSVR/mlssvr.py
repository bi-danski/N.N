class MLSSVR:
    def __init__(self, C, epsilon, gamma, layers):
        self.layers = layers
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
