import numpy as np
from matplotlib import pyplot as plt
import pyttsx3

engine = pyttsx3.init()

# Flower data [length, width, colour (1 = red, 2 = blue)]
data = [[3,   1.5, 1],
        [2,   1,   0],
        [4,   1.5, 1],
        [3,   1,   0],
        [3.5, 0.5, 1],
        [2,   0.5, 0],
        [5.5, 1,   1],
        [1,   1,   0]]

unknown_f = [4.5, 1]


def sigmoid(x):
    return 1/(1 + np.exp(-x))


def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))


def which_flower(length, width):
    z = length * w1 + width * w2 + b
    pred = round(sigmoid(z))
    if pred == 0:
        engine.say("The flower is definitely blue")
        engine.runAndWait()
        print("blue")
    else:
        engine.say('The flower is definitely red')
        engine.runAndWait()
        print("red")


# Graphing sigmoid and its derivative
T = np.linspace(-10, 10, 100)
plt.plot(T, sigmoid(T), c='r')
plt.plot(T, sigmoid_prime(T), c='b')

# plt.show()


# # Scatter Data
# plt.axis([0, 6, 0, 6])
# plt.grid()
# for i in range(len(data)):
#     point = data[i]
#     color = "r"
#     if point[2] == 0:
#         color = "b"
#     plt.scatter(point[0], point[1], c=color)
# plt.show()


# Training Loop
learning_rate = 0.2
costs = []

w1 = np.random.randn()
w2 = np.random.randn()
b = np.random.randn()

for i in range(500000):
    ri = np.random.randint(len(data))
    point = data[ri]
    length = point[0]
    width = point[1]

    z = length * w1 + width * w2 + b
    pred = sigmoid(z)
    target = point[2]
    cost = np.square(pred - target)

    costs.append(cost)

    # Derivatives:
    dcost_dpred = 2 * (pred - target)
    dpred_dz = sigmoid_prime(z)
    dz_dw1 = length
    dz_dw2 = width
    dz_db = 1
    # CHAIN RULE (dy/dx  =  dy/du  * du/dx)
    dcost_dw1 = dcost_dpred * dpred_dz * dz_dw1
    dcost_dw2 = dcost_dpred * dpred_dz * dz_dw2
    dcost_b = dcost_dpred * dpred_dz * dz_db

    # Updating weights and bias
    w1 = w1 - learning_rate * dcost_dw1
    w2 = w2 - learning_rate * dcost_dw2
    b = b - learning_rate * dcost_b

plt.plot(costs)
# plt.show()

# Prints predictions and answers side-by-side
for i in range(len(data)):
    point = data[i]
    # print(point)
    length = point[0]
    width = point[1]
    z = length * w1 + width * w2 + b
    pred = sigmoid(z)
    # print("prediction: {}".format(round(pred)))

unknown_length = unknown_f[0]
unknown_width = unknown_f[1]
which_flower(unknown_length, unknown_width)

