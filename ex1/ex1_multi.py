import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
from numpy.linalg import inv


def compute_cost(X, Y, theta):
    m = X.shape[0]
    return 1 / (2 * m) * np.sum(np.square(np.dot(X, theta) - Y))


def gradient_descent(X, Y, theta, alpha, num_iters):
    m = X.shape[0]
    J_history = np.zeros((num_iters, 1))
    for x in range(0, num_iters):
        theta = theta - np.multiply((alpha / m), np.dot(X.T, np.dot(X, theta) - Y))
        J_history[x, :] = compute_cost(X, Y, theta)
    return (theta, J_history)


def features_normalize(X):
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0, ddof=1)
    X_norm = np.divide(np.subtract(X, mu), sigma)
    return (X_norm, mu, sigma)

data = genfromtxt('ex1data2.txt', delimiter=',')
X = data[:, 0:2]
Y = data[:, 2]
m = Y.shape[0]
Y = np.reshape(Y, (m, 1))

print('First 10 examples from the dataset\n')
print(np.column_stack((X[0:10, :], Y[0:10, :])))

(X, mu, sigma) = features_normalize(X)
X = np.column_stack((np.ones(m), X))

# Run gradient descent
alpha = 0.05
num_iters = 150
theta = np.zeros((3, 1))
(theta, J_history) = gradient_descent(X, Y, theta, alpha, num_iters)

(theta1, J1) = gradient_descent(X, Y, theta, 0.01, 50);
(theta2, J2) = gradient_descent(X, Y, theta, 0.03, 50);
(theta3, J3) = gradient_descent(X, Y, theta, 0.1, 50);
(theta4, J4) = gradient_descent(X, Y, theta, 0.3, 50);
(theta5, J5) = gradient_descent(X, Y, theta, 1, 50);

# Plot the convergence graph
plt.plot(np.arange(0, J_history.size, 1), J_history, '-b', lw=2)
plt.xlabel('Number of iterations')
plt.ylabel('Cost J')
plt.show()

plt.figure()
plt.plot(np.arange(0, 50, 1), J1[0:50], 'b')
plt.plot(np.arange(0, 50, 1), J2[0:50], 'r')
plt.plot(np.arange(0, 50, 1), J3[0:50], 'g')
plt.plot(np.arange(0, 50, 1), J4[0:50], 'y')
plt.plot(np.arange(0, 50, 1), J5[0:50], 'k')
plt.xlabel('Number of iterations')
plt.ylabel('Cost J')
plt.show()

print('theta value obtained from gradient descent')
print(theta)

#Predict the price of a 1650 sq-ft, 3 bedroom house
price  = np.dot(np.column_stack((1, np.divide(np.subtract(np.array([[1650., 3.]]), mu), sigma))), theta).item()
print('Predicted price of a 1650 sq-ft, 3 br house using gradient descent: %.2f: $' %price)

#Using normal equation
X = data[:, 0:2]
X = np.column_stack((np.ones(m), X))
theta_normeq = np.dot(np.dot(inv(np.dot(X.T, X)),X.T),Y)
print('theta value obtained using normal equation')
print(theta_normeq)

#Predict the price of a 1650 sq-ft, 3 bedroom house
price  = np.dot(np.column_stack((1, np.array([[1650, 3]]))), theta_normeq).item()
print('Predicted price of a 1650 sq-ft, 3 br house using gradient descent: %.2f: $' %price)