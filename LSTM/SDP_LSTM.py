import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from bayes_opt import BayesianOptimization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt


X = pd.read_csv("data/training.csv")
y = pd.read_csv("data/validation.csv")
X = X.values
y = y.values

X = np.nan_to_num(X, nan=np.nanmean(X))
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


def data_augmentation(X_train, y_train):
    noise_factor = 0.05
    noise = np.random.normal(loc=0.0, scale=noise_factor, size=X_train.shape)
    X_train_augmented = X_train + noise

    X_train = np.concatenate((X_train, X_train_augmented), axis=0)
    y_train = np.concatenate((y_train, y_train), axis=0)

    return X_train, y_train


X_train, y_train = data_augmentation(X_train, y_train)

X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))


def create_lstm_model(input_shape, units=16, l2_reg=0.001):
    model = keras.Sequential()
    model.add(layers.LSTM(units, input_shape=input_shape, return_sequences=True, kernel_regularizer=l2(l2_reg), recurrent_dropout=0.2))
    model.add(layers.LSTM(units, kernel_regularizer=l2(l2_reg), recurrent_dropout=0.2))
    model.add(layers.Dense(y_train.shape[1]))
    return model


def train_lstm_model(X_train, y_train, X_val, y_val, epochs=700, batch_size=10, l2_reg=0.001, learning_rate=0.001, units=16):
    input_shape = X_train.shape[1:]
    lstm_model = create_lstm_model(input_shape, units=units, l2_reg=l2_reg)
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    lstm_model.compile(optimizer=optimizer, loss='mse')

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, min_delta=1e-4)

    history = lstm_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size,
                             callbacks=[early_stopping])
    return lstm_model, history


def evaluate_model_bayesian(units, l2_reg, learning_rate, batch_size, epochs):
    units = int(units)
    batch_size = int(batch_size)
    epochs = int(epochs)
    
    lstm_model = create_lstm_model(X_train.shape[1:], units=units, l2_reg=l2_reg)
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    lstm_model.compile(optimizer=optimizer, loss='mse')

    lstm_model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    y_pred = lstm_model.predict(X_val)
    mse = mean_squared_error(y_val, y_pred)
    return -mse


def optimize_parameters_bayesian(X_train, y_train, X_val, y_val):
    pbounds = {
        'units': (16, 256),
        'l2_reg': (1e-5, 0.05),
        'learning_rate': (1e-5, 0.05),
        'batch_size': (1, 100),
        'epochs': (10, 1000)
    }

    optimizer = BayesianOptimization(
        f=evaluate_model_bayesian,
        pbounds=pbounds,
        random_state=42
    )

    optimizer.maximize(init_points=5, n_iter=10) 
    best_params = optimizer.max['params']
    return best_params


def adjust_parameters_bayesian(best_params, X_train, y_train, X_val, y_val):
    units = int(best_params['units'])
    l2_reg = best_params['l2_reg']
    learning_rate = best_params['learning_rate']
    batch_size = int(best_params['batch_size'])
    epochs = int(best_params['epochs'])

    lstm_model = create_lstm_model(X_train.shape[1:], units=units, l2_reg=l2_reg)
    lstm_model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss='mse')

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, min_delta=1e-4)

    lstm_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size, callbacks=[early_stopping])

    return lstm_model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, mae, r2


def calculate_performance_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mse, mae, r2


def cross_validate(X, y, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    mse_scores = []
    mae_scores = []
    r2_scores = []

    for train_index, val_index in kf.split(X):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
        X_val = X_val.reshape((X_val.shape[0], 1, X_val.shape[1]))

        lstm_model, _ = train_lstm_model(X_train, y_train, X_val, y_val, 
                                         epochs=initial_params['epochs'], 
                                         batch_size=initial_params['batch_size'], 
                                         l2_reg=initial_params['l2_reg'], 
                                         learning_rate=initial_params['learning_rate'], 
                                         units=initial_params['units'])
        y_pred = lstm_model.predict(X_val)

        mse, mae, r2 = calculate_performance_metrics(y_val, y_pred)
        mse_scores.append(mse)
        mae_scores.append(mae)
        r2_scores.append(r2)

    avg_mse = np.mean(mse_scores)
    avg_mae = np.mean(mae_scores)
    avg_r2 = np.mean(r2_scores)

    return avg_mse, avg_mae, avg_r2


def evaluate_final_model(lstm_model, X_test, y_test):
    y_pred = lstm_model.predict(X_test)
    mse, mae, r2 = calculate_performance_metrics(y_test, y_pred)
    return mse, mae, r2


initial_params = {
    'batch_size': 45,
    'epochs': 937,
    'l2_reg': 0.017377007516895342,
    'learning_rate': 0.013342706781993856,
    'units': 54
}


X_train_main, X_val, y_train_main, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

if __name__ == "__main__":
    print("[*]Training initial LSTM model with initial parameters")
    lstm_model, history = train_lstm_model(X_train_main, y_train_main, X_val, y_val, 
                                           epochs=initial_params['epochs'], 
                                           batch_size=initial_params['batch_size'], 
                                           l2_reg=initial_params['l2_reg'], 
                                           learning_rate=initial_params['learning_rate'],
                                           units=initial_params['units'])
    initial_metrics = evaluate_model(lstm_model, X_test, y_test)
    print(f"Initial model performance metrics: MSE={initial_metrics[0]}, MAE={initial_metrics[1]}, R2={initial_metrics[2]}")

    print("[*]Performing Bayesian optimization")
    best_params = optimize_parameters_bayesian(X_train_main, y_train_main, X_val, y_val)
    print(f"Best parameters found: {best_params}")

    print("[*]Adjusting parameters based on Bayesian optimization")
    lstm_model = adjust_parameters_bayesian(best_params, X_train_main, y_train_main, X_val, y_val)

    print("[*]Performing cross-validation...")
    avg_metrics = cross_validate(X_scaled, y)
    print(f"Cross-validation metrics: MSE={avg_metrics[0]}, MAE={avg_metrics[1]}, R2={avg_metrics[2]}")

    print("[*]Evaluating the final model on test set...")
    performance_metrics = evaluate_final_model(lstm_model, X_test, y_test)
    print(f"Test set performance metrics: MSE={performance_metrics[0]}, MAE={performance_metrics[1]}, R2={performance_metrics[2]}")

    lstm_model.save_model("SlurryDeformationPrediction_LSTM.h5")
    lstm_model.save_weights("weights.h5")

    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Learning Curves')
    plt.legend()
    plt.savefig('Learning_Curves.png')
    plt.show()

    y_pred = lstm_model.predict(X_test)
    plt.scatter(y_test, y_pred, label='Predictions')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Predicted vs Actual Values')
    plt.legend()
    plt.savefig('Predicted_vs_Actual_Values.png')
    plt.show()

