import pickle

file_name = 'nn.pkl'
with open(file_name, 'rb') as file:
    net = pickle.load(file)
    print("Successfully loaded")