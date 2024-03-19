import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("data.csv")

# Feature Scaling 
# Z-score normalization
data["year_scaled"] = (data["year"] - data["year"].mean()) / data["year"].std()
data["income_scaled"] = (data["per capita income (US$)"] - data["per capita income (US$)"].mean()) / data["per capita income (US$)"].std()

def cost(training_set, w, b):
    m = len(training_set)
    x = training_set["year_scaled"].values
    y = training_set["income_scaled"].values

    f_wb = w * x + b
    j_wb = ((f_wb - y) ** 2).sum() / (2 * m)  # Calculate mean squared error

    return j_wb


def gradient_descent(training_set, w_now, b_now, alpha):
    m = len(training_set)
    x = training_set["year_scaled"].values
    y = training_set["income_scaled"].values
    
    f_wb = w_now * x + b_now
    dj_dw = np.dot(f_wb - y, x) / m
    dj_db = np.sum(f_wb - y) / m

    w = w_now - alpha * dj_dw
    b = b_now - alpha * dj_db

    return w, b


# Initialize weights with small values
w = 0.01
b = 0.01

alpha = 0.1
epochs = 1000
iterations = 0

cost_values = []  # List to store cost values

for i in range(epochs):
    w, b = gradient_descent(data, w, b, alpha)

    j_wb = cost(data, w, b)
    cost_values.append(j_wb)  # Append cost value for plotting
    
    iterations = i

    # Check convergence
    if i > 0 and math.isclose(cost_values[-1], cost_values[-2], rel_tol=1e-20):
        print("Total iterations:", iterations)
        break


# Plotting the cost function
# visualizing gradient descent is converging
plt.plot(range(0, iterations + 1), cost_values, color="red")
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Cost Function")
plt.show()

# Predictions for the original data scale
data["predictions"] = w * data["year_scaled"] + b
data["predictions"] = data["predictions"] * data["per capita income (US$)"].std() + data["per capita income (US$)"].mean()

# Plotting the data and regression line
plt.scatter(data["year"], data["per capita income (US$)"], color="black", label="Actual Data")
plt.plot(data["year"], data["predictions"], color="red", label="Regression Line")
plt.xlabel("Year")
plt.ylabel("Per Capita Income (US$)")
plt.title("Linear Regression using Gradient Descent")
plt.legend()
plt.show()

# Assuming you want to predict for the year 2020
new_year = 2020

# Scaling the new year value using the same scaling factors
scaled_new_year = (new_year - data["year"].mean()) / data["year"].std()

# Using the learned weights to make predictions
prediction_scaled = w * scaled_new_year + b

# Inverse scaling to get the prediction in the original scale
prediction = prediction_scaled * data["per capita income (US$)"].std() + data["per capita income (US$)"].mean()

print(f"Predicted Per Capita Income for the year {new_year}: ${prediction:.2f}")
