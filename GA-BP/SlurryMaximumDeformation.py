import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from deap import base, creator, tools, algorithms
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


def create_bp_model(input_shape, hidden_units, activation='relu', l2_reg=0.001):
    model = keras.Sequential()
    model.add(layers.Input(shape=(input_shape,)))
    for units in hidden_units:
        if units > 0:
            model.add(layers.Dense(units, activation=activation, kernel_regularizer=l2(l2_reg)))
        else:
            break
    model.add(layers.Dense(y_train.shape[1]))  # Output layer with 94 units
    return model


def train_bp_model(X_train, y_train, X_val, y_val, epochs=1000, batch_size=32, l2_reg=0.001):
    input_shape = X_train.shape[1]
    hidden_units = [32, 16]
    bp_model = create_bp_model(input_shape, hidden_units, l2_reg=l2_reg)
    bp_model.compile(optimizer='adam', loss='mse')

    # Early Stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

    history = bp_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size,
                           callbacks=[early_stopping])
    return bp_model, history


def evaluate_individual(individual):
    input_shape = X_train.shape[1]
    hidden_units = [max(1, int(individual[0])), max(1, int(individual[1]))]
    activation = 'relu'
    learning_rate = abs(individual[2])
    batch_size = max(1, int(abs(individual[3])))
    epochs = max(1, int(abs(individual[4])))
    l2_reg = abs(individual[5])

    bp_model = create_bp_model(input_shape, hidden_units, activation, l2_reg=l2_reg)
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    bp_model.compile(optimizer=optimizer, loss='mse')

    bp_model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    y_pred = bp_model.predict(X_val)
    mse = mean_squared_error(y_val, y_pred)
    return mse,


def optimize_parameters(population_size=50, generations=20):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("attr_int", np.random.randint, 16, 128)
    toolbox.register("attr_float", np.random.uniform, 0.0001, 0.1)
    toolbox.register("individual", tools.initCycle, creator.Individual,
                     (toolbox.attr_int, toolbox.attr_int, toolbox.attr_float, toolbox.attr_int, toolbox.attr_int,
                      toolbox.attr_float), n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", evaluate_individual)

    population = toolbox.population(n=population_size)
    algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2, ngen=generations, verbose=True)
    best_individual = tools.selBest(population, k=1)[0]
    return best_individual


def integrate_ga_with_bp(best_individual):
    input_shape = X_train.shape[1]
    hidden_units = [max(1, int(best_individual[0])), max(1, int(best_individual[1]))]
    activation = 'relu'
    learning_rate = abs(best_individual[2])
    batch_size = max(1, int(abs(best_individual[3])))
    epochs = max(1, int(abs(best_individual[4])))
    l2_reg = abs(best_individual[5])

    optimized_model = create_bp_model(input_shape, hidden_units, activation, l2_reg=l2_reg)
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    optimized_model.compile(optimizer=optimizer, loss='mse')

    return optimized_model, epochs, batch_size


def train_ga_bp_model(bp_model, X_train, y_train, X_val, y_val, epochs, batch_size, l2_reg=0.001):
    # Early Stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

    bp_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size,
                 callbacks=[early_stopping])


def initial_optimization(X_train, y_train, X_val, y_val):
    best_individual = optimize_parameters()
    bp_model, epochs, batch_size = integrate_ga_with_bp(best_individual)
    train_ga_bp_model(bp_model, X_train, y_train, X_val, y_val, epochs, batch_size)
    return bp_model, best_individual


def cross_validate(X, y, n_splits=2):
    kf = KFold(n_splits=n_splits)
    metrics = []
    for train_index, val_index in kf.split(X):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        bp_model, best_individual = initial_optimization(X_train, y_train, X_val, y_val)
        train_ga_bp_model(bp_model, X_train, y_train, X_val, y_val, 60, 32)

        y_pred = bp_model.predict(X_val)
        mse, mae, r2 = calculate_performance_metrics(y_val, y_pred)
        metrics.append((mse, mae, r2))

    avg_metrics = np.mean(metrics, axis=0)
    return avg_metrics


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, mae, r2


def adjust_parameters(bp_model, best_individual, X_train, y_train, X_val, y_val):
    mse, mae, r2 = evaluate_model(bp_model, X_val, y_val)
    if mse > 0.01:
        # Reduce learning rate
        best_individual[2] *= 0.5
        bp_model, epochs, batch_size = integrate_ga_with_bp(best_individual)
        train_ga_bp_model(bp_model, X_train, y_train, X_val, y_val, epochs, batch_size)
    return bp_model


def evaluate_final_model(model, X_test, y_test):
    performance_metrics = evaluate_model(model, X_test, y_test)
    return performance_metrics


def calculate_performance_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mse, mae, r2


X_train_main, X_val, y_train_main, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

if __name__ == "__main__":
    print("[*]Training BP model")
    bp_model, history = train_bp_model(X_train_main, y_train_main, X_val, y_val)

    print("[*]Optimizing parameters using GA")
    best_individual = optimize_parameters()

    print("[*]Integrating GA with BP model")
    bp_model, epochs, batch_size = integrate_ga_with_bp(best_individual)

    print("[*]Training GA-BP model")
    train_ga_bp_model(bp_model, X_train_main, y_train_main, X_val, y_val, epochs, batch_size)

    print("[*]Performing cross-validation...")
    avg_metrics = cross_validate(X_scaled, y)
    print(f"Cross-validation metrics: MSE={avg_metrics[0]}, MAE={avg_metrics[1]}, R2={avg_metrics[2]}")

    print("[*]Evaluating the final model on test set...")
    performance_metrics = evaluate_final_model(bp_model, X_test, y_test)
    print(
        f"Test set performance metrics: MSE={performance_metrics[0]}, MAE={performance_metrics[1]}, R2={performance_metrics[2]}")

    print("[*]Adjusting parameters based on validation set...")
    bp_model = adjust_parameters(bp_model, best_individual, X_train_main, y_train_main, X_val, y_val)

    print("[*]Re-evaluating the final model on test set...")
    performance_metrics = evaluate_final_model(bp_model, X_test, y_test)
    print(
        f"Test set performance metrics (after adjustment): MSE={performance_metrics[0]}, MAE={performance_metrics[1]}, R2={performance_metrics[2]}")

    print("[*]Saving Model...")
    bp_model.save("SlurryDeformationPrediction.h5")
    bp_model.save_weights("weights.h5")

    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Learning Curves')
    plt.legend()
    plt.savefig('Learning_Curves.png')
    plt.show()

    y_pred = bp_model.predict(X_test)
    plt.scatter(y_test, y_pred, label='Predictions')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Predicted vs Actual Values')
    plt.legend()
    plt.savefig('Predicted_vs_Actual_Values.png')
    plt.show()
