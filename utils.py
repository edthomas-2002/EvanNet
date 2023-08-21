import math
import numpy as np

def sigmoid(x):
    # input dimensions: m rows = 1 per input, n cols per row = num neurons in output layer
    transferred = []
    for row in x:
        row_transfer = [1.0 / (1.0 + math.exp(-n_o)) for n_o in row]
        transferred.append(row_transfer)
    return np.asarray(transferred)

def sigmoid_derivative(x):
    # input dimensions: m rows = 1 per input, n cols per row = num neurons in output layer
    derivs = []
    for row in x:
        row_derivs = [(1.0 / (1.0 + math.exp(-n_o))) * (1 - (1.0 / (1.0 + math.exp(-n_o)))) for n_o in row]
        derivs.append(row_derivs)
    return np.asarray(derivs)

def softmax(x):
    # input dimensions: m rows = 1 per input, n cols per row = num neurons in output layer
    max_x = np.max(x, axis=1, keepdims=True)
    e_x = np.exp(x - max_x)
    sum_e_x = np.sum(e_x, axis=1, keepdims=True)
    softmax_values = e_x / sum_e_x
    return softmax_values
    
def softmax_derivative(y):
    # input dimensions: m rows = 1 per input, n cols per row = num neurons in output layer
    num_examples, num_classes = y.shape
    d_softmax = np.zeros((num_examples, num_classes, num_classes))
    for i in range(num_examples):
        for j in range(num_classes):
            for k in range(num_classes):
                if j == k:
                    d_softmax[i, j, k] = y[i, j] * (1 - y[i, j])
                else:
                    d_softmax[i, j, k] = -y[i, j] * y[i, k]
    return d_softmax

def ReLU(x):
    # input dimensions: m rows = 1 per input, n cols per row = num neurons in output layer
    transferred = []
    for row in x:
        row_transfer = [max(0, n_o) for n_o in row]
        transferred.append(row_transfer)
    return np.asarray(transferred)

def ReLU_derivative(x):
    # input dimensions: m rows = 1 per input, n cols per row = num neurons in output layer
    derivs = []
    for row in x:
        row_derivs = [1 if n_o > 0 else 0 for n_o in row]
        derivs.append(row_derivs)
    return np.asarray(derivs)

def identity(x):
    # input dimensions: m rows = 1 per input, n cols per row = num neurons in output layer
    return x

def identity_derivative(x):
    # input dimensions: m rows = 1 per input, n cols per row = num neurons in output layer
    return np.ones([len(x), len(x[0])], dtype=int)

def BCE_loss(y, p):
    # input dimensions: m rows = 1 per input, n cols per row = num neurons in output layer
    # BCE is good if ONLY 1 output neuron
    p = [pred - 10 ** -9 if pred[0] == 1 else pred + 10 ** -9 if pred[0] == 0 else pred for pred in p]
    loss = []
    for r in range(len(y)):
        loss.append(- (y[r][0] * math.log(p[r][0]) + (1 - y[r][0]) * math.log(1 - p[r][0])))
    return np.asarray(loss)
   
def BCE_derivative(y, p):
    # input dimensions: m rows = 1 per input, n cols per row = num neurons in output layer
    # BCE is good if ONLY 1 output neuron
    p = [pred - 10 ** -9 if pred[0] == 1 else pred + 10 ** -9 if pred[0] == 0 else pred for pred in p]
    derivs = []
    for r in range(len(y)):
        derivs.append([-((y[r][0] / p[r][0]) - ((1 - y[r][0]) / (1 - p[r][0])))])
    return np.asarray(derivs)

def CCE_loss(y, p):
    # NEED one-hot encoding of y -- [0,1,0,0,0] or [0,0,0,1,0], etc
    # input dimensions: m rows = 1 per input, n cols per row = num neurons in output layer ie num categories
    epsilon = 1e-15  # Small constant to avoid log(0)
    p = np.clip(p, epsilon, 1 - epsilon)  # Clip predictions to avoid log(0) or log(1)
    loss = -np.sum(y * np.log(p)) / y.shape[0]
    return loss
   
def CCE_derivative(y, p):
    # input dimensions: m rows = 1 per input, n cols per row = num neurons in output layer
    epsilon = 1e-15  # Small constant to avoid division by zero
    p = np.clip(p, epsilon, 1 - epsilon)  # Clip predictions to avoid division by zero
    d_loss_d_y_pred = - (y / p) / y.shape[0]
    return d_loss_d_y_pred
    p += 1 * (10 ** -9)
    return - y / p

def MSE_loss(y, p):
    # input dimensions: m rows = 1 per input, n cols per row = num neurons in output layer
    running_sum = [0 for n in y[0]]
    for r in range(len(y)):
        this_row = (y[r] - p[r]) ** 2
        running_sum += this_row
    return running_sum / len(y)
   
def MSE_derivative(y, p):
    # input dimensions: m rows = 1 per input, n cols per row = num neurons in output layer
    derivs = []
    for r in range(len(y)):
        derivs.append(2 * (p[r] - y[r]))
    return np.asarray(derivs)
