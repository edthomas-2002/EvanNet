import csv, random
import numpy as np
from utils import *
import pickle

data = []
with open('data.csv', mode ='r')as file:
   csvFile = csv.reader(file)
   for line in csvFile:
      row = []
      for elt in line:
         row.append(int(elt))
      row = np.asarray(row)
      data.append(row)
train_data = np.asarray(data[:int(0.75 * len(data))])
test_data = np.asarray(data[int(0.75 * len(data)):])


class NN:
   
   def __init__(self, structure): # assumes every neuron connected to each neuron in next layer
      self.structure = structure
      self.num_layers = len(structure)
      self.weights = [] # a list of np arrays
      self.biases = []
      self.y_dim = self.structure[-1]
      for l in range(1, self.num_layers):
         ws = self.initialize_weights(structure[l-1], structure[l])
         bs = self.initialize_biases(structure[l], 1)
         self.weights.append(ws)
         self.biases.append(bs)

   def initialize_biases(self, input_size, output_size, std_dev=0.01):
      biases = np.random.normal(loc=0.0, scale=std_dev, size=(input_size, output_size))
      return_list = []
      for elt in biases:
         return_list.append(elt[0])
      return np.asarray(return_list)

   def initialize_weights(self, input_size, output_size, std_dev=0.01):
      weights = np.random.normal(loc=0.0, scale=std_dev, size=(input_size, output_size))
      return weights

   def loss(self, actual_y, predicted):
      # dimensions: m rows = 1 per input, n cols per row = num neurons in output layer
      # BCE is good if ONLY 1 output neuron
      return CCE_loss(actual_y, predicted)
   
   def loss_derivative(self, actual_y, predicted):
      return CCE_derivative(actual_y, predicted)

   def transfer(self, activations): 
      # activations: output matrix resulting from rows input matrix matmul w/ weights matrix
      # dimensions: m rows = 1 per input, n cols per row = num neurons in output layer
      return ReLU(activations)
   
   def output_transfer(self, activations): 
      return softmax(activations)
   
   def transfer_deriv(self, activations): 
      return ReLU_derivative(activations)
   
   def output_transfer_deriv(self, activations): 
      return softmax_derivative(activations)
      
   def forward_pass(self, rows):
      input = rows
      for l in range(len(self.weights)):
         input = np.matmul(input, self.weights[l])
         input += self.biases[l]
         if l == len(self.weights) - 1: # bc last transfer is diff
            input = self.output_transfer(input)
         else:
            input = self.transfer(input)
      return input

   def back_prop(self, rows, target_outputs, learning_rate=0.01): 
      # forward prop
      input = rows
      weighted_sums = [] # num elts = num layers, num elts in each elt is num neurons in that layer
      outputs = [rows]
      for l in range(len(self.weights)):
         input = np.matmul(input, self.weights[l])
         input += self.biases[l]
         weighted_sums.append(input)
         if l == len(self.weights) - 1: # bc last transfer is diff
            input = self.output_transfer(input)
         else:
            input = self.transfer(input)
         outputs.append(input)

      # calculate loss
      loss = self.loss(target_outputs, outputs[-1])
      print(np.mean(loss))

      pb_structure = [el for el in self.structure]
      pb_structure.append(1)
      passback = [] # pb[l][i][j] get us what we need for dL_d_output_i
      for l in range(len(self.weights)):
         layer = []
         for j in range(pb_structure[l+1]):
            initial = []
            for k in range(pb_structure[l+2]):
               jk = [1 for row in rows]
               initial.append(jk)
            layer.append(initial)
         passback.append(layer)
      derivs = self.loss_derivative(target_outputs, outputs[-1])
      for j in range(self.structure[-1]):
         passback[-1][j][0] = derivs.swapaxes(0,1)[j]
      # The gradient tells us the direction of steepest increase in func, eg (3,4,5) take 3 steps x, 4 y, 5 z
      # so we want to move opposite this: -gradient
      # For each neuron: new_ws = old_ws - l_r * gradient.
      # These gradients are averaged over the mini-batch to get an estimate 
      # of the overall gradient for that mini-batch
      # d_loss_d_weight_ij = d_loss_d_output_j * d_output_j_d_z_j * a_i
      for l in range(1, len(self.weights)+1):
         if l == 1: # bc diff transfer for output layer
            layer_activation_derivs = self.output_transfer_deriv(weighted_sums[-l])
         else:
            layer_activation_derivs = self.transfer_deriv(weighted_sums[-l])
         for i in range(self.structure[-l-1]):
            for j in range(self.structure[-l]):
               components = passback[-l][j] # components has as many elts as j has destination neurons
               d_loss_d_output_j = 0
               for partial in components:
                  d_loss_d_output_j += partial
               d_output_j_d_input_j = layer_activation_derivs.swapaxes(0,1)[j]
               if l == 1:
                  # If output_transfer is softmax:  -- note: want result of this 
                  # operation to be a vector
                  summed = [0 for r in range(len(rows))]
                  for n in range(self.structure[-1]):
                     summed += layer_activation_derivs[:, n, j]
                  d_output_j_d_input_j = summed
               a_i = outputs[-l-1].swapaxes(0,1)[i]
               if l < len(self.weights): # bc don't want to try to pass back from 1st hidden layer
                  passback[-l-1][i][j] = d_loss_d_output_j * d_output_j_d_input_j * self.weights[-l][i][j]
               d_loss_d_weight_ij = d_loss_d_output_j * d_output_j_d_input_j * a_i
               d_loss_d_weight_ij = np.mean(d_loss_d_weight_ij)
               self.weights[-l][i][j] -= learning_rate * d_loss_d_weight_ij
               if i == 0: # bc only need to do this once per j
                  d_loss_d_bias_j = d_loss_d_output_j * d_output_j_d_input_j
                  d_loss_d_bias_j = np.mean(d_loss_d_bias_j)
                  self.biases[-l][j] -= learning_rate * d_loss_d_bias_j

   def train(self, data, n_epochs, learning_rate=0.01, batch_size=50):
      for it in range(n_epochs):
         random.shuffle(data)
         X = np.asarray([row[:-self.y_dim] for row in data])
         Y = np.asarray([row[-self.y_dim:] for row in data])
         for b in range(int(len(X)/batch_size)):
            batch_x = X[b * batch_size : (b + 1) * batch_size]
            batch_y = Y[b * batch_size : (b + 1) * batch_size]
            for r in range(len(batch_x)):
               self.back_prop(batch_x, batch_y, learning_rate)

   def test(self, data):
      X = np.asarray([row[:-self.y_dim] for row in data])
      Y = np.asarray([row[-self.y_dim:] for row in data])
      output = self.forward_pass(X)
      output = np.round(output)
      outpt_classes = np.argmax(output, axis=1)
      Y_classes = np.argmax(Y, axis=1)
      return np.sum(outpt_classes == Y_classes) / Y.shape[0]


net = NN([3,2,4])
net.train(train_data, 5, batch_size=200)
result = net.test(test_data)
print("res",result)
