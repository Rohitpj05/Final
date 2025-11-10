import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Load the data
df = pd.read_csv(r'C:\Users\rohit\OneDrive\Desktop\Sem 5\3.End_Sem ML\temperatures.csv')

# -----------------------------------------------
# Model 1: Predicting 'JAN'
# -----------------------------------------------

# Define features (X) and target (y)
x = df[['YEAR']]
y = df[['JAN']]

# Split the data into training and testing sets
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3)

# Create and train the model
model = LinearRegression()
model.fit(xtrain, ytrain)

# Print the model's intercept and slope
print(f"Intercept: {model.intercept_}")
print(f"Slope: {model.coef_}")

# Use the model to make predictions on the test data
prediction = model.predict(xtest)

# --- THIS IS THE CORRECTED SECTION ---
#
# The error metrics must compare the actual test values (ytest)
# with the values your model predicted (prediction).
#
print("--- Model Evaluation ---")
print("Mean Absolute Error:", metrics.mean_absolute_error(ytest, prediction))
print("Mean Squared Error:", metrics.mean_squared_error(ytest, prediction))
print("Root Mean Squared Error:", np.sqrt(metrics.mean_squared_error(ytest, prediction)))
#
# --------------------------------------

# Plot the results
plt.figure()
plt.title("JAN Temperature vs. Year")
plt.scatter(x, y, color='g', label='Actual Data')
plt.plot(x, model.predict(x), color='k', label='Regression Line')
plt.xlabel('Year')
plt.ylabel('JAN Temperature')
plt.legend()
plt.show()

# You would then repeat this entire process for 'MAR', 'APR', 'JUN', and 'MAY',
# ensuring you always use (ytest, prediction) in your metrics.
