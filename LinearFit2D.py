import numpy as np
import matplotlib.pyplot as plt
# Generating random 2D data
np.random.seed(150)
X = np.random.normal(5, 2, 100)
Y = 5 * X + np.random.normal(6, 4, 100)

# Plotting the original data
plt.scatter(X, Y, color='blue')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Original Data')
plt.grid(True)
plt.show()
# Step 2: Defining the functions for mean squared error and gradient descent

# Function to compute mean squared error
def compute_mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Gradient descent algorithm for linear regression
def gradient_descent(X, Y, learning_rate, iterations):
    n = len(Y)
    slope = 0
    intercept = 0
    
    for _ in range(iterations):
        predictions = intercept + slope * X
        
        slope_gradient = -2/n * np.sum((Y - predictions) * X)
        intercept_gradient = -2/n * np.sum(Y - predictions)
        
        slope -= learning_rate * slope_gradient
        intercept -= learning_rate * intercept_gradient
    
    return intercept, slope
# Step 3: Apply gradient descent and plot the best fit line
# Parameters
learning_rate = 0.0001
iterations = 1000

# Applying gradient descent
intercept, slope = gradient_descent(X, Y, learning_rate, iterations)

# Predicted values
predicted_values = intercept + slope * X

# Plotting the original data and the linear fit
plt.scatter(X, Y, color='blue', label='Original Data')
plt.plot(X, predicted_values, color='red', label='Best Linear Fit')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Linear Regression with Gradient Descent')
plt.legend()
plt.grid(True)
plt.show()
# Step 4: Output coefficients and mean squared error

# Outputting coefficients and mean squared error
print("Intercept (b):", intercept)
print("Slope (m):", slope)
print("Mean Squared Error:", compute_mse(Y, predicted_values))
