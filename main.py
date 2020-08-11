import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import os

from model import Model
from layers import *
from activations import *
from losses import *
from optimizers import *
from accuracies import *

# Loads a mnist dataset
def load_mnist_dataset(dataset, path):
    
    # Scan all the directories and create a list of labels
    labels = os.listdir(os.path.join(path, dataset))

    # Create lists for samples and labels
    X = []
    y = []

    # For each label folder
    for label in labels:
        # And for each image in given folder
        for file in os.listdir(os.path.join(path, dataset, label)):
            # Read the image
            image = cv2.imread(os.path.join(path, dataset, label, file), cv2.IMREAD_UNCHANGED)

            # And append it and a label to the lists
            X.append(image)
            y.append(label)    


    # Convert the data to proper numpy arrays and return
    return np.array(X), np.array(y).astype('uint8')


# MNIST dataset (train + test)
def create_data_mnist(path):

    # Load both sets separately
    X, y = load_mnist_dataset('train', path)
    X_test, y_test = load_mnist_dataset('test', path)

    # And return all the data
    return X, y, X_test, y_test

X, y, X_test, y_test = create_data_mnist('fashion_mnist_images')
keys = np.array(range(X.shape[0]))
np.random.shuffle(keys)
X = X[keys]
y = y[keys]

X = (X.reshape(X.shape[0], -1).astype(np.float16)) / 255
X_test = (X_test.reshape(X_test.shape[0], -1).astype(np.float16)) / 255

y = X.copy()
y_test = X_test.copy()

base = int(input('Base neuron amount?')) # 512
iterations = int(input('Iterations?')) # 4
dropout = float(input('Dropout? 0-1')) # 0.1

epochs = int(input('epochs?'))

initial_activation = input('Initial activation? sigmoid/relu')
final_activation = input('Final activation? sigmoid/relu/linear')

model = Model()

model.add(Layer_Dense(X.shape[1], base))
model.add(Activation_Sigmoid())

print(f'Added layer {X.shape[1]} -> {base}')
for i in range(0, iterations):
    model.add(Layer_Dense(int(base / 2**i), int(base / 2**(i+1))))
    if initial_activation == 'sigmoid':
        model.add(Activation_Sigmoid())
    elif initial_activation == 'relu':
        model.add(Activation_ReLU())
    print(f'Added layer {int(base / 2**i)} -> {int(base / 2**(i+1))}')

    if i != iterations - 1:
        print(f'{dropout} dropout')
        model.add(Layer_Dropout(dropout))

for i in range(0, iterations):
    model.add(Layer_Dense(int(base / 2**(iterations-i)), int(base / 2**(iterations-i-1))))
    if initial_activation == 'sigmoid':
        model.add(Activation_Sigmoid())
    elif initial_activation == 'relu':
        model.add(Activation_ReLU())
    print(f'Added layer {int(base / 2**(iterations-i))} -> {int(base / 2**(iterations-i-1))}')

    # if i != 0:
    #     print(f'{dropout} dropout')
    #     model.add(Layer_Dropout(dropout))

print(f'Added layer {base} -> {X.shape[1]}')
print()

model.add(Layer_Dense(base, X.shape[1]))
if final_activation == 'sigmoid':
    model.add(Activation_Sigmoid())
elif final_activation == 'relu':
    model.add(Activation_ReLU())
elif final_activation == 'linear':
    model.add(Activation_Linear())

model.set(
    loss=Loss_MeanSquaredError(),
    optimizer=Optimizer_Adam(decay=1e-9),
    accuracy=Accuracy_Categorical()
)

model.finalize()
model.train(X, y, validation_data=(X_test, y_test), epochs=epochs, batch_size=256, print_every=100)

for data in X:
    original = (data.reshape((28, 28)) * 255 ).astype(np.uint32)
    
    predicted_data = model.forward(data, False)
    predicted = (predicted_data.reshape((28, 28)) * 255).astype(np.uint32)

    plt.figure()
    plt.imshow(original, cmap='gray')
    plt.figure()
    plt.imshow(predicted, cmap='gray')

    plt.show()