import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def compute_cost(x, y, w_now, b_now):
    x = x.values
    y = y.values

    m = x.shape[0]
    w_now = np.array(w_now)

    j_wb = 0

    for i in range(m):
        f_wb_i = np.dot(w_now, x[i]) + b_now
        j_wb += (f_wb_i - y[i]) ** 2

    j_wb /= (2 * m)

    return j_wb


def gradient_descent(x, y, w_now, b_now, alpha):
    x = x.values
    y = y.values
    
    m, n = x.shape

    dj_dw = np.zeros((n,))
    dj_db = 0.0

    for i in range(m):
        f_wb = np.dot(w_now, x[i]) + b_now
        dj_dw += (f_wb - y[i]) * x[i]
        dj_db += f_wb - y[i]
    
    dj_dw /= m
    dj_db /= m

    w = w_now - alpha * dj_dw        
    b = b_now - alpha * dj_db  

    return w, b


# create target data
x = np.arange(0, 20, 1)
y = x**2

# engineer features
df = pd.DataFrame({"x": x, "x^2": x**2, "x^3": x**3, "y": y})

X_train = df.drop(columns="y")
Y_train = df[["y"]]

b = 0
w = [0, 0, 0]
alpha = 0.0000001
epochs = 10000
iterations = 0

cost_values = []  # List to store cost values

for i in range(epochs):
    w, b = gradient_descent(X_train, Y_train, w, b, alpha)

    j_wb = compute_cost(X_train, Y_train, w, b)
    cost_values.append(j_wb)  # Append cost value for plotting
    
    iterations = i

    # Check convergence
    if i > 0 and abs(cost_values[-1] - cost_values[-2]) < 1e-3:
        print("Converged at iteration:", iterations)
        break

    
# Format each value of w individually to 2 decimal points
formatted_w = np.array([f'{val:.2f}' for val in w])

print("Formatted w:", formatted_w)
print("b:", b)


plt.scatter(x, y, marker='x', c='r', label="Actual Value");
plt.title("x, x**2, x**3 features")
plt.plot(x, np.dot(X_train, w) + b, label="Predicted Value"); 
plt.xlabel("x"); 
plt.ylabel("y"); 
plt.legend(); 
plt.show()
