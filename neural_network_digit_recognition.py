import numpy as np
import pandas as pd
from scipy.special import expit

# Load datasets
mnist_dataset_train = pd.read_csv("mnist_train.csv")
mnist_dataset_test = pd.read_csv("mnist_test.csv")

# Extract labels
labels_train = mnist_dataset_train.pop("label")
labels_test = mnist_dataset_test.pop("label")

# Convert testing data to numpy array
testing_data = mnist_dataset_test.values

# Class attributes
epoch = 20
learning_rate = 0.01  # Adjust the learning rate as needed
batch_size = 1000  # Adjust the batch size as needed

# Define network architecture
input_size = 784
hidden_size = 100  # Increase the number of hidden units for better representation
output_size = 10

# Initialize weights and biases
weights_input_hidden = np.random.randn(input_size, hidden_size) * 0.01
weights_hidden_output = np.random.randn(hidden_size, output_size) * 0.01
bias_hidden = np.zeros((1, hidden_size))
bias_output = np.zeros((1, output_size))

# Activation function (sigmoid)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Forward pass function
def forward_pass(input_data):
    hidden_input = np.dot(input_data, weights_input_hidden) + bias_hidden
    hidden_output = sigmoid(hidden_input)
    output = np.dot(hidden_output, weights_hidden_output) + bias_output
    output = expit(output)  # Using expit as sigmoid for numerical stability
    return hidden_output, output

# Backpropagation function
def backpropagation(input_data, hidden_output, output, labels):
    delta_output = output - np.eye(output_size)[labels]
    delta_hidden = np.dot(delta_output, weights_hidden_output.T) * hidden_output * (1 - hidden_output)

    w_gradient_hidden_output = np.dot(hidden_output.T, delta_output)
    w_gradient_input_hidden = np.dot(input_data.T, delta_hidden)

    b_gradient_output = np.sum(delta_output, axis=0)
    b_gradient_hidden = np.sum(delta_hidden, axis=0)

    return w_gradient_input_hidden, w_gradient_hidden_output, b_gradient_hidden, b_gradient_output

# Gradient descent function
def gradient_descent(w_gradient_input_hidden, w_gradient_hidden_output, b_gradient_hidden, b_gradient_output):
    global weights_input_hidden, weights_hidden_output, bias_hidden, bias_output
    weights_input_hidden -= learning_rate * w_gradient_input_hidden
    weights_hidden_output -= learning_rate * w_gradient_hidden_output
    bias_hidden -= learning_rate * b_gradient_hidden
    bias_output -= learning_rate * b_gradient_output

# Evaluation function for testing
def evaluate_testing(epoch):
    correct = 0
    for i in range(len(testing_data)):
        _, output = forward_pass(testing_data[i:i+1])
        if np.argmax(output) == labels_test[i]:
            correct += 1
    accuracy = correct / len(testing_data) * 100
    print(f"Epoch {epoch + 1}: {correct}/{len(testing_data)} ({accuracy:.2f}% accuracy)")

# Training loop
for epoch in range(epoch):
    for i in range(0, len(mnist_dataset_train), batch_size):
        batch_data = mnist_dataset_train.iloc[i:i+batch_size].values
        batch_labels = labels_train.iloc[i:i+batch_size].values

        hidden_output, output = forward_pass(batch_data)
        w_gradient_input_hidden, w_gradient_hidden_output, b_gradient_hidden, b_gradient_output = backpropagation(
            batch_data, hidden_output, output, batch_labels)
        gradient_descent(w_gradient_input_hidden, w_gradient_hidden_output, b_gradient_hidden, b_gradient_output)

    evaluate_testing(epoch)
