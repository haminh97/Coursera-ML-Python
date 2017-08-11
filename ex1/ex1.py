import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


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

data = genfromtxt('ex1data1.txt', delimiter=',')

X = data[:, 0]
Y = data[:, 1]
m = X.shape[0]
X = np.reshape(X, (m, 1))
Y = np.reshape(Y, (m, 1))

# Plot the data
plt.figure()
plt.plot(X, Y, 'rx', ms=10)
plt.xlabel('Population of city in 10000s')
plt.ylabel('Profit in $10000')
plt.show()

# Cost and gradient descent
X = np.column_stack((np.ones(m), X))
theta = np.zeros((2, 1))

print('With theta = [0 ; 0]\nCost computed = %f\n' % compute_cost(X, Y, theta))
print('Expected cost value (approx) 32.07\n')

print('\nWith theta = [-1 ; 2]\nCost computed = %f\n' % compute_cost(X, Y, np.array([[-1], [2]])))
print('Expected cost value (approx) 54.24\n');

# Gradient Descent
print('\nRunning Gradient Descent ...\n')
iterations = 1500
alpha = 0.01
theta = gradient_descent(X, Y, theta, alpha, iterations)[0]

# Plot the Linear regression
plt.figure()
plt.plot(X[:, 1], Y, 'rx', ms=10)
plt.plot(X[:, 1], np.dot(X, theta), '-')
plt.xlabel('Population of city in 10000s')
plt.ylabel('Profit in $10000')
plt.legend(['Training data', 'Linear regression'])
plt.show()

predict1 = np.dot([[1, 3.5]], theta).item()
print('For population = 35,000, we predict a profit of %.2f\n' % (predict1 * 10000))
predict2 = np.dot([[1, 7]], theta).item()
print('For population = 70,000, we predict a profit of %.2f\n' % (predict2 * 10000))

# Visualizing Theta

theta0_vals = np.linspace(-10, 10, 100)
theta1_vals = np.linspace(-1, 4, 100)
J_vals = np.zeros((theta0_vals.size, theta1_vals.size))

for i in range(0, theta0_vals.size):
    for j in range(0, theta1_vals.size):
        t = np.array([[theta0_vals[i]], [theta1_vals[j]]])
        J_vals[i][j] = compute_cost(X, Y, t)

J_vals = J_vals.T
theta0_vals = np.reshape(theta0_vals, (theta0_vals.size, 1))
theta1_vals = np.reshape(theta1_vals, (theta1_vals.size, 1))

# Surface plot
fig = plt.figure()
ax = fig.gca(projection='3d')
theta0_vals, theta1_vals = np.meshgrid(theta0_vals, theta1_vals)
ax.plot_surface(theta0_vals, theta1_vals, J_vals, cmap=cm.coolwarm)
plt.xlabel(r'$\theta_0$')
plt.ylabel(r'$\theta_1$')
plt.show()

# Contour plot
plt.figure()
plt.contour(theta0_vals, theta1_vals, J_vals, np.logspace(-2, 3, 20))
plt.plot(theta[0], theta[1], 'rx', ms=10, lw=2)
plt.xlabel(r'$\theta_0$')
plt.ylabel(r'$\theta_1$')
plt.show()
