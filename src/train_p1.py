import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt


path_to_csv = "../Adam_Raslan.csv"

def mean_absolute_error(actual_values, predicted_values):
    if len(actual_values) != len(predicted_values):
        raise ValueError("Length of actual values and predicted values must be the same.")

    absolute_errors = [abs(actual - predicted) for actual, predicted in zip(actual_values, predicted_values)]
    mae = sum(absolute_errors) / len(absolute_errors)
    return mae
def calculate_rmse(y_true, y_pred):
    mse = np.mean((y_true - y_pred)**2)
    rmse = np.sqrt(mse)
    return rmse

def calculate_mape(y_true, y_pred):
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return mape

def parse_activities(file_path):
    df = pd.read_csv(file_path)
    df.dropna(inplace=True)
    predictions = df["average_speed"].values

    for index, row in df.iterrows():
        if row['type'] != "Run" or row['has_heartrate'] == "FALSE":
            df.drop(index, inplace=True)
    df.drop(columns=["name", "distance", "sport_type", "type", "has_heartrate"], inplace=True)
    # Convert 'start_date' column to datetime
    df['start_date'] = pd.to_datetime(df['start_date'])
    df['year'] = df['start_date'].dt.year
    df['month'] = df['start_date'].dt.month
    df['day'] = df['start_date'].dt.day
    # One-hot encode year
    df['year'] = df['year'].astype(str)
    df = pd.get_dummies(df, columns=['year'], prefix='year')

    # One-hot encode month
    df['month'] = df['month'].astype(str)
    df = pd.get_dummies(df, columns=['month'], prefix='month')

    # One-hot encode day
    df.drop(columns=["start_date"], inplace=True)
    return df

def lin_reg(X_train, y_train, alpha =0.1):
    # Initialize Linear Regression model
    model = Ridge(alpha=alpha)

    # Train the model
    model.fit(X_train, y_train)

    return model

def random_forest_reg(X_train, y_train, n_estimators=100, max_depth=None):
    # Initialize Random Forest Regression model
    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)

    # Train the model
    model.fit(X_train, y_train)

    return model


def find_optimal_alpha_in_lr(X_train, y_train, X_dev, y_dev):
    alpha_values = [0.001, 0.01, 0.1, 1, 10]  # Different alpha values to try
    dev_mapes = []

    for alpha in alpha_values:
        lr_model = lin_reg(X_train, y_train, alpha=alpha)
        dev_predictions = lr_model.predict(X_dev)
        dev_mape = calculate_mape(y_dev, dev_predictions)
        dev_mapes.append(dev_mape)
        print("Alpha:", alpha)
        print("MAPE on Dev Set:", dev_mape)
        print("")

    # Plot the results
    plt.plot(alpha_values, dev_mapes, marker='o')
    plt.title('MAPE vs Alpha for Linear Regression')
    plt.xlabel('Alpha')
    plt.ylabel('MAPE on Development Set')
    plt.xscale('log')  # Use logarithmic scale for better visualization if alpha values vary widely
    plt.grid(True)
    plt.show()


def find_opt_RF(X_train, y_train, X_dev, y_dev):
    n_estimators_values = [10, 50, 100, 150, 200]  # Different values for the number of decision trees
    depth_values = [5, 10, 15, 20, 25]  # Different values for the maximum depth of each tree
    dev_mapes_n_estimators = []

    # Optimal Number of Decision Trees and Depth
    min_estimator = 10
    min_depth = 5
    min_dev_mape = 100
    for n_estimators in n_estimators_values:
        for depth in depth_values:
            rf_model = random_forest_reg(X_train, y_train, n_estimators=n_estimators, max_depth = depth)
            dev_predictions = rf_model.predict(X_dev)
            dev_mape = calculate_mape(y_dev, dev_predictions)
            dev_mapes_n_estimators.append(dev_mape)
            print("Number of Decision Trees:", n_estimators)
            print("MAPE on Dev Set:", dev_mape)
            print("")
            if dev_mape < min_dev_mape:
                min_estimator = n_estimators
                min_depth = depth
                min_dev_mape = dev_mape
    print("This is min_dev_mape:", min_dev_mape)
    print(min_depth)
    print(min_estimator)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(dev_mapes_n_estimators)), dev_mapes_n_estimators, label='MAPE vs. (Number of Decision Trees,Depth)')
    plt.xlabel('Configuration')
    plt.ylabel('MAPE')
    plt.title('MAPE vs. Configuration')
    plt.legend()
    plt.xticks(ticks=range(len(dev_mapes_n_estimators)),
               labels=[f'{n}, {d}' for n in n_estimators_values for d in depth_values], rotation=45)
    plt.tight_layout()
    plt.show()

def main():
    file_path = "../Adam_Raslan_activities.csv"
    df = parse_activities(file_path)
    y_train = df["average_speed"]
    df.drop(columns="average_speed", inplace=True)
    X_train = df.values

    # Split the data into training and temporary data (80% training, 20% temp)
    X_train, X_temp, y_train, y_temp = train_test_split(X_train, y_train, test_size=0.3, random_state=42)

    # Split the temporary data into development and testing sets (50% each of the temp data)
    X_dev, X_test, y_dev, y_test = train_test_split(X_temp, y_temp, test_size=0.66, random_state=42)
    # find_optimal_alpha_in_lr(X_train, y_train, X_dev, y_dev)
    # find_opt_RF(X_train, y_train, X_dev, y_dev)

    lr_model = lin_reg(X_train, y_train)
    dev_predictions = lr_model.predict(X_dev)
    print("LINEAR REGRESSION RESULTS: ")
    test_predictions = lr_model.predict(X_test)
    dev_mape = calculate_mape(y_dev, dev_predictions)
    print("MAPE on Dev Set:", dev_mape)

    test_mape = calculate_mape(y_test, test_predictions)
    print("MAPE on Test Set:", test_mape)

    print("")
    print("RANDOM FOREST REGRESSION RESULTS: ")
    rf_model = random_forest_reg(X_train, y_train, 50, 10)
    dev_predictions = rf_model.predict(X_dev)
    test_predictions = rf_model.predict(X_test)
    dev_mape = calculate_mape(y_dev, dev_predictions)
    print("MAPE on Dev Set:", dev_mape)

    test_mape = calculate_mape(y_test, test_predictions)
    print("MAPE on Test Set:", test_mape)

    for pred1, pred2 in zip(list(dev_predictions), list(y_dev)):
        print(str(pred1) + " " + str(pred2))


main()