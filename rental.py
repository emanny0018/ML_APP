# rental_price_prediction.py

# IMPORTING LIBRARIES AND ALGORITHMS FOR TRAINING
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# LOADING THE DATA
rentalPD = pd.read_csv('/mlapp/data/london_processed_data.csv')

# Check data to see if correct
print(rentalPD.head(5))

# DATA PREPARATION NOTE: TRAINING DATA (X), EXPECTED PREDICTION (Y)
X = rentalPD[['BATHROOMS', 'SIZE']].values  # Features: number of bathrooms and square footage
Y = rentalPD[['rent']].values  # Predicting label against X as rent price

# SPLIT TEST 20%, TRAIN 80%
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

# MODEL TRAINING
model = LinearRegression().fit(X_train, y_train)
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# PREDICTING RENT PRICES IN LONDON, ENGLAND
# Replace Apartment_sqft with your actual new data
def user_input():
    rooms = float(input("What is the number of rooms? "))
    sq_ft = float(input("What is the square footage you desire? "))
    return [[rooms, sq_ft]]
Apartment_sqft = user_input()  # Example features: BATHROOMS = 2, SIZE = 861
predicted_rent = model.predict(Apartment_sqft)

# PREDICTED PRICE FOR RENT BASED ON DATA ENTERED
print(f"Predicted rent for new data: £{predicted_rent[0][0]}")

# Checking difference between predicted and actual price
print("This is the difference between Actual and Predicted Price", y_test[0][0] - predicted_rent[0][0])

# CALCULATE METRICS
# Calculate MSE for training and test sets
mse_train = mean_squared_error(y_train, y_train_pred)
mse_test = mean_squared_error(y_test, y_test_pred)

# Calculate R-squared for training and test sets
r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)

print(f"Mean Squared Error (train): {mse_train}")
print(f"Mean Squared Error (test): {mse_test}")
print(f"R-squared (train): {r2_train}")
print(f"R-squared (test): {r2_test}")

# Calculate RMSE for training and test data
rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
print("Training Data Root Mean Squared Error (RMSE):", rmse_train)

rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
print("Test Data Root Mean Squared Error (RMSE):", rmse_test)

# Calculate loss (difference between actual and predicted rent) for both training and test data
loss_train = y_train - y_train_pred
loss_test = y_test - y_test_pred

# Randomly select 5 errors from the training data
train_sample_indices = np.random.choice(len(loss_train), 5, replace=False)
train_sample_errors = loss_train[train_sample_indices]

# Randomly select 5 errors from the test data
test_sample_indices = np.random.choice(len(loss_test), 5, replace=False)
test_sample_errors = loss_test[test_sample_indices]

print("########################################### Training Data Rental Price Prediction Error ###########################################")
print("Sample Loss (Difference between actual and predicted rent) for Training Data:")
print(train_sample_errors)
print("########################################### Training Data Rental Price Prediction Error ###########################################")

print("########################################### Test Data Rental Price Prediction Error ###########################################")
print("Sample Loss (Difference between actual and predicted rent) for Test Data:")
print(test_sample_errors)
print("########################################### Test Data Rental Price Prediction Error ###########################################")

# Define a tolerance for the prediction accuracy
tolerance = 500  # e.g., predictions within 500 units of the actual value

# Calculate accuracy for training data
train_accuracy = np.mean(np.abs(loss_train) <= tolerance)
print("Training Data Accuracy within tolerance:", train_accuracy)

# Calculate accuracy for test data
test_accuracy = np.mean(np.abs(loss_test) <= tolerance)
print("Test Data Accuracy within tolerance:", test_accuracy)

# Calculate percentage difference for training data
percentage_diff_train = np.abs((y_train - y_train_pred) / y_train) * 100

# Calculate mean percentage difference for training data
mean_percentage_diff_train = np.mean(percentage_diff_train)
print("Mean Percentage Difference for Training Data:", mean_percentage_diff_train, "%")

# Calculate percentage difference for test data
percentage_diff_test = np.abs((y_test - y_test_pred) / y_test) * 100

# Calculate mean percentage difference for test data
mean_percentage_diff_test = np.mean(percentage_diff_test)
print("Mean Percentage Difference for Test Data:", mean_percentage_diff_test, "%")

# Randomly select 5 percentage differences from the training data
train_sample_percentage_diff = percentage_diff_train[train_sample_indices]

# Randomly select 5 percentage differences from the test data
test_sample_percentage_diff = percentage_diff_test[test_sample_indices]

print("########################################### Training Data Rental Price Prediction Percentage Difference ###########################################")
print("Sample Percentage Difference between actual and predicted rent for Training Data:")
print(train_sample_percentage_diff)
print("########################################### Training Data Rental Price Prediction Percentage Difference ###########################################")

print("########################################### Test Data Rental Price Prediction Percentage Difference ###########################################")
print("Sample Percentage Difference between actual and predicted rent for Test Data:")
print(test_sample_percentage_diff)
print("########################################### Test Data Rental Price Prediction Percentage Difference ###########################################")

print("Execution Completed")

