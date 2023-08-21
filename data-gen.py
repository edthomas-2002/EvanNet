import random, csv
import numpy as np

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def convert_to_one_hot(labels, num_cats):
    ret = []
    for elt in labels:
        one_hot = [0 for i in range(num_cats)]
        one_hot[elt] = 1
        ret.append(one_hot)
    return np.asarray(ret)

def simple_binary(size):
    arr = []
    for i in range(size):
        choice = random.randint(0,1)
        arr.append([choice, choice])
    return arr

def AND():
    return [[0,0,0], [0,1,0], [1,0,0], [1,1,1]]

def arithmetic_class(op, size):
    arr = []
    for i in range(size):
        f = random.randint(1,500)
        s = random.randint(1,500)
        if op == 0:
            sol = f + s
        elif op == 1:
            sol = f - s
        elif op == 2:
            sol = f * s
        elif op == 3:
            sol = f // s
        sample = [f, s, sol, 0, 0, 0, 0]
        sample[3 + op] = 1
        arr.append(sample)
    return arr

def times_2(size):
    return [[1,2], [2,4], [3,6], [4,8]]
    arr = []
    for i in range(size):
        f = random.randint(1,499)
        sample = [f, f*2]
        arr.append(sample)
    return arr

def calculator(size):
    arr = []
    for i in range(size):
        f = random.randint(0,499)
        s = random.randint(0,499)
        sample = [f, s, f+s]
        arr.append(sample)
    return arr

plus_arr = arithmetic_class(0,2000)
minus_arr = arithmetic_class(1,2000)
mul_arr = arithmetic_class(2,2000)
div_arr = arithmetic_class(3,2000)
combined = plus_arr + minus_arr + mul_arr + div_arr
random.shuffle(combined)

# bin = simple_binary(10000)

# addition = calculator(20000)

#two = times_2(500)

# train_batch_name = "cifar-10-batches-py/data_batch_1"
# test_batch_name = "cifar-10-batches-py/test_batch"
# cifar_dict = unpickle(train_batch_name)
# cifar = cifar_dict[b'data']
# labels = convert_to_one_hot(cifar_dict[b'labels'],10)
# full_cifar = []
# for r in range(len(cifar)):
#     full_cifar.append(np.concatenate((cifar[r], labels[r])))
# full_cifar = np.asarray(full_cifar)

dest_file = "data.csv"
with open(dest_file, 'w') as csvfile: 
    csvwriter = csv.writer(csvfile) 
    csvwriter.writerows(combined)
