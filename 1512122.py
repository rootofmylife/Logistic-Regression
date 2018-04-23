import numpy as np 
import csv
import json
import matplotlib.pyplot as plt

def sigmoid(x, derivative=False):
    return x*(1-x) if derivative else 1/(1+np.exp(-x))

def computeCost(data, theta, ld, yl):
    sum = 0
    for i in range(len(data)):
        sum += -yl[i] * np.log(sigmoid(data[i].dot(theta))) - (1 - yl[i]) * np.log(1 - sigmoid(data[i].dot(theta)))
    sum /= len(data)

    sum_theta = 0
    for j in theta:
        sum_theta += j * j
    sum_theta = (sum_theta * ld) / (2 * len(data))

    return sum + sum_theta

def computeGradient(data, theta, ld, yl, ap):
    for i in range(len(theta)):
        sum = 0
        for j in range(len(data)):
            sum += (sigmoid(data[j].dot(theta)) - yl[j]) * data[j][i]

        sum /= len(data)
        if i == 0:
            theta[i] -= ap * sum
        else:
            theta[i] -= (ap * sum) + (ld / len(data)) * theta[i]
    return theta

def gradientDescent(data, theta, ld, yl, ap, numIter):
    for i in range(numIter):
        theta = computeGradient(data, theta, ld, yl, ap)

    return theta

def predict(new_x, new_y, theta):
    predict_data = mapFeature(new_x, new_x)
    point = []
    for i in predict_data:
        point.append(sigmoid(i.dot(theta)))
    return point
        

def mapFeature(X1, X2):
# MAPFEATURE Feature mapping function to polynomial features
#
#   MAPFEATURE(X1, X2) maps the two input features
#   to quadratic features used in the regularization exercise.
#
#   Returns a new feature array with more features, comprising of 
#   X1, X2, X1.^2, X2.^2, X1*X2, X1*X2.^2, etc..
#
#   Inputs X1, X2 must be the same size
    degree = 6
    out = np.ones([len(X1), (degree+1)*(degree+2) // 2], dtype = float)
    idx = 1
    for i in range(1, degree+1):
        for j in range(0, i+1):
            a1 = X1 ** (i-j)
            a2 = X2 ** j
            out[:, idx] = a1*a2
            idx += 1
    return out

if __name__ == "__main__":
    with open('config.json') as configFile:
        config = json.load(configFile)

    with open(config['Dataset'], 'rb') as csvfile:
        data = np.loadtxt(csvfile, delimiter=",")
        new_data = mapFeature(data[:,0], data[:,1])

    lambda_value = config['Lambda']
    alpha = config['Alpha']
    numiter = config['NumIter']
    theta = config['Theta']
    y_label = data[:,2]

    theta = gradientDescent(new_data, theta, lambda_value, y_label, alpha, numiter)

    output = {"Theta" : theta, "Cost" : computeCost(new_data, theta, lambda_value, y_label)}

    with open('model.json', 'w') as outputFile:
        json.dump(output, outputFile)

    x0_1 = np.linspace(-2, 2, num=50)
    x0_2 = np.linspace(-2, 2, num=50)
    graph = mapFeature(x0_1, x0_2)
    y0 = graph * theta
    plt.plot(data[:,0], data[:,1], 'ro')
    plt.axis([-2, 2, -2, 2])
    plt.plot(graph, y0)
    plt.grid(True)
    plt.show()