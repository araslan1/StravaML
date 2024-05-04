import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import matplotlib.pyplot as plt


path_to_csv = "../data/Adam_Raslan_activities.csv"

helper_plot = True

def calculate_mape(y_true, y_pred):
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return mape

def parse_activities(file_path):
    df = pd.read_csv(file_path)
    df.dropna(inplace=True)
    predictions = df["average_speed"].values
    for index, row in df.iterrows():
        if row['type'] != "Run" or row['has_heartrate'] == "FALSE" or row['average_speed'] > 11:
            df.drop(index, inplace=True)
    df.drop(columns=["name", "distance", "sport_type", "type", "has_heartrate"], inplace=True)
    # Convert 'start_date' column to datetime
    df['start_date'] = pd.to_datetime(df['start_date'])
    df['year'] = df['start_date'].dt.year
    df['month'] = df['start_date'].dt.month
    df['day'] = df['start_date'].dt.day
    df['time'] = df['start_date'].dt.hour
    # One-hot encode year
    df['year'] = df['year'].astype(str)
    df = pd.get_dummies(df, columns=['year'], prefix='year')

    # One-hot encode month
    df['month'] = df['month'].astype(str)
    df = pd.get_dummies(df, columns=['month'], prefix='month')

    # One-hot encode day
    df.drop(columns=["start_date"], inplace=True)
    return df

def lin_reg(X_train, y_train, alpha = 1):
    # Initialize Linear Regression model
    model = Ridge(alpha=alpha)

    # Train the modelIssaquah?
    model.fit(X_train, y_train)

    return model

def random_forest_reg(X_train, y_train, n_estimators=100, max_depth=5):
    # Initialize Random Forest Regression model
    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)

    # Train the model
    model.fit(X_train, y_train)

    return model

def gradient_boosting_reg(X_train, y_train, n_estimators=100, max_depth=5, learning_rate = 0.1):
    # Initialize Random Forest Regression model
    model = GradientBoostingRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, random_state=42,validation_fraction=0.1,
    n_iter_no_change=20)

    # Train the model
    model.fit(X_train, y_train)

    return model


def find_optimal_alpha_in_lr(X_train, y_train, X_dev, y_dev):
    alpha_values = [0, 0.001, 0.01, 0.1, 1, 10]  # Different alpha values to try
    dev_mapes = []

    for alpha in alpha_values:
        lr_model = lin_reg(X_train, y_train, alpha=alpha)
        dev_predictions = lr_model.predict(X_dev)
        dev_mape = calculate_mape(y_dev, dev_predictions)
        dev_mapes.append(dev_mape)
    min_mape_index = dev_mapes.index(min(dev_mapes))

    # Return the optimal alpha value
    # Plot the results
    if helper_plot:
        plt.plot(alpha_values, dev_mapes, marker='o')
        plt.title('DEV MAPE vs Alpha for Linear Regression')
        plt.xlabel('Alpha')
        plt.ylabel('MAPE on Development Set')
        plt.xscale('log')  # Use logarithmic scale for better visualization if alpha values vary widely
        plt.grid(True)
        plt.show()
    return alpha_values[min_mape_index]


def find_opt_RF(X_train, y_train, X_dev, y_dev):
    n_estimators_values = [10, 50, 100, 150, 200]  # Different values for the number of decision trees
    depth_values = [5, 10, 15, 20, 25]  # Different values for the maximum depth of each tree
    dev_mapes = []

    # Optimal hyperparameters
    min_estimator = None
    min_depth = None
    min_dev_mape = float('inf')

    for n_estimators in n_estimators_values:
        for depth in depth_values:
                rf_model = RandomForestRegressor(n_estimators=n_estimators, max_depth=depth, random_state=42)
                rf_model.fit(X_train, y_train)
                dev_predictions = rf_model.predict(X_dev)
                dev_mape = calculate_mape(y_dev, dev_predictions)
                dev_mapes.append(dev_mape)
                if dev_mape < min_dev_mape:
                    min_estimator = n_estimators
                    min_depth = depth

                    min_dev_mape = dev_mape
    if helper_plot:
        plt.figure(figsize=(10, 6))
        plt.plot(range(len( dev_mapes)),  dev_mapes,
                 label='DEV MAPE vs. (Number of Decision Trees,Depth)')
        plt.xlabel('Configuration')
        plt.ylabel('MAPE')
        plt.title('MAPE vs. Configuration')
        plt.legend()
        plt.xticks(ticks=range(len(dev_mapes)),
                   labels=[f'{n}, {d}' for n in n_estimators_values for d in depth_values], rotation=45)
        plt.tight_layout()
        plt.show()
    return [min_estimator, min_depth]
def find_opt_GF(X_train, y_train, X_dev, y_dev):
    n_estimators_values = [10, 50, 100, 150, 200]  # Different values for the number of decision trees
    depth_values = [5, 10, 15, 20, 25]  # Different values for the maximum depth of each tree
    learning_rate_values = [0.01, 0.05, 0.1, 1, 0.5]  # Different values for the learning rate
    dev_mapes = []

    # Optimal hyperparameters
    min_estimator = None
    min_depth = None
    min_learning_rate = None
    min_dev_mape = float('inf')

    for n_estimators in n_estimators_values:
        for depth in depth_values:
            for learning_rate in learning_rate_values:
                rf_model = GradientBoostingRegressor(n_estimators=n_estimators, max_depth=depth, learning_rate=learning_rate,random_state=42)
                rf_model.fit(X_train, y_train)
                dev_predictions = rf_model.predict(X_dev)
                dev_mape = calculate_mape(y_dev, dev_predictions)
                dev_mapes.append(dev_mape)
                if dev_mape < min_dev_mape:
                    min_estimator = n_estimators
                    min_depth = depth
                    min_learning_rate = learning_rate
                    min_dev_mape = dev_mape
    # Plotting

    print(min_estimator)
    print(min_depth)
    print(min_learning_rate)
    if helper_plot:
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(dev_mapes)), dev_mapes, label='DEV MAPE vs. (Number of Decision Trees, Depth, Learning Rate)')
        plt.xlabel('Configuration')
        plt.ylabel('MAPE')
        plt.title('MAPE vs. Configuration')
        plt.legend()
        plt.xticks(ticks=range(len(dev_mapes)),
                   labels=[f'({n}, {d}, {lr})' for n in n_estimators_values for d in depth_values for lr in learning_rate_values],
                   rotation=45)
        plt.tight_layout()
        plt.show()
    return [min_estimator, min_depth, min_learning_rate]


def linear_reg_report(X_train, y_train, X_dev, X_test, y_dev, y_test, optimal_alpha):

    print("Number of training samples: " + str(X_train.shape[0]))
    print("Number of test samples: " + str(X_test.shape[0]))
    print("numbers of dev samples: " + str(X_dev.shape[0]))
    print("Number of features: " + str(X_dev.shape[1]))
    lr_model = lin_reg(X_train, y_train, optimal_alpha)

    train_predictions = lr_model.predict(X_train)
    train_mape = calculate_mape(y_train, train_predictions)
    print("MAPE on Training Set:", train_mape)

    dev_predictions = lr_model.predict(X_dev)
    print("LINEAR REGRESSION RESULTS: ")
    dev_mape = calculate_mape(y_dev, dev_predictions)
    print("MAPE on Dev Set:", dev_mape)

    test_predictions = lr_model.predict(X_test)
    test_mape = calculate_mape(y_test, test_predictions)
    print("MAPE on Test Set:", test_mape)

    # Calculate MSE for linear regression
    mse_dev_lr = mean_squared_error(y_dev, dev_predictions)
    print("MSE on Dev Set (Linear Regression):", mse_dev_lr)

    mse_test_lr = mean_squared_error(y_test, test_predictions)
    print("MSE on Test Set (Linear Regression):", mse_test_lr)


def random_forest_report(X_train, y_train, X_dev, X_test, y_dev, y_test, optimum):
    print("RANDOM FOREST REGRESSION RESULTS: ")
    rf_model = random_forest_reg(X_train, y_train, optimum[0], optimum[1])

    train_predictions = rf_model.predict(X_train)
    train_mape = calculate_mape(y_train, train_predictions)
    print("MAPE on Training Set:", train_mape)

    dev_predictions = rf_model.predict(X_dev)
    test_predictions = rf_model.predict(X_test)
    dev_mape = calculate_mape(y_dev, dev_predictions)
    print("MAPE on Dev Set:", dev_mape)

    test_mape = calculate_mape(y_test, test_predictions)
    print("MAPE on Test Set:", test_mape)

    # Calculate MSE for random forest regression
    mse_dev_rf = mean_squared_error(y_dev, dev_predictions)
    print("MSE on Dev Set (Random Forest Regression):", mse_dev_rf)

    mse_test_rf = mean_squared_error(y_test, test_predictions)
    print("MSE on Test Set (Random Forest Regression):", mse_test_rf)
    return test_predictions


def gradient_boosting_report(X_train, y_train, X_dev, X_test, y_dev, y_test, optimum):
    print("GRADIENT BOOSTING REGRESSION RESULTS: ")
    gf_model = gradient_boosting_reg(X_train, y_train, optimum[0], optimum[1], optimum[2])

    train_predictions = gf_model.predict(X_train)
    train_mape = calculate_mape(y_train, train_predictions)
    print("MAPE on Training Set:", train_mape)

    dev_predictions = gf_model.predict(X_dev)
    test_predictions = gf_model.predict(X_test)
    dev_mape = calculate_mape(y_dev, dev_predictions)
    print("MAPE on Dev Set:", dev_mape)

    test_mape = calculate_mape(y_test, test_predictions)
    print("MAPE on Test Set:", test_mape)

    # Calculate MSE for random forest regression
    mse_dev_rf = mean_squared_error(y_dev, dev_predictions)
    print("MSE on Dev Set (Random Forest Regression):", mse_dev_rf)

    mse_test_rf = mean_squared_error(y_test, test_predictions)
    print("MSE on Test Set (Random Forest Regression):", mse_test_rf)

def pretty_print_predictions(dev_predictions, y_dev):
    for pred1, pred2 in zip(list(dev_predictions), list(y_dev)):
        decimal_part = pred1 - int(pred1)
        seconds = decimal_part*60
        decimal_part_2 = pred2 - int(pred2)
        seconds_2 = decimal_part_2 * 60
        print("Model Prediction: " + str(int(pred1)) + " minutes " +
              str(int(seconds)) + " seconds, Actual Pace: " + str(int(pred2)) + " minutes " + str(int(seconds_2)) + " seconds")

def main():
    file_path = path_to_csv
    df = parse_activities(file_path)
    y_train = df["average_speed"]
    df.drop(columns="average_speed", inplace=True)
    X_train = df.values

    # Split the data into training and temporary data (80% training, 20% temp)
    X_train, X_temp, y_train, y_temp = train_test_split(X_train, y_train, test_size=0.3, random_state=42)

    # Split the temporary data into development and testing sets (50% each of the temp data)
    X_dev, X_test, y_dev, y_test = train_test_split(X_temp, y_temp, test_size=0.66, random_state=42)
    optimal_alpha = find_optimal_alpha_in_lr(X_train, y_train, X_dev, y_dev)
    optimal_forest = find_opt_RF(X_train, y_train, X_dev, y_dev)
    optimal_gradient_boost = find_opt_GF(X_train, y_train, X_dev, y_dev)


    linear_reg_report(X_train, y_train, X_dev, X_test, y_dev, y_test, optimal_alpha)
    print("")
    test_predictions = random_forest_report(X_train, y_train, X_dev, X_test, y_dev, y_test, optimal_forest)
    print("")
    gradient_boosting_report(X_train, y_train, X_dev, X_test, y_dev, y_test, optimal_gradient_boost)
    pretty_print_predictions(test_predictions, y_test)


main()