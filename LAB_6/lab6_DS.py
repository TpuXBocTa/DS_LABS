import numpy as np


X = np.array([
    [0, 1, 1],
    [0, 0, 0],
    [1, 0, 0],
    [1, 0, 1],
    [0, 0, 1]
], dtype=float)

Y = np.array([
    [0],
    [0],
    [1],
    [1],
    [1]
], dtype=float)


def sigmoid(value):
    return 1.0 / (1.0 + np.exp(-value))


def sigmoid_derivative(value):
    return value * (1.0 - value)


rng = np.random.default_rng(12)

input_neurons = 3
hidden_neurons = 4
output_neurons = 1

W1 = rng.normal(0.0, 0.5, size=(input_neurons, hidden_neurons))
b1 = np.zeros((1, hidden_neurons))

W2 = rng.normal(0.0, 0.5, size=(hidden_neurons, output_neurons))
b2 = np.zeros((1, output_neurons))

learning_rate = 0.8
epochs = 12000
eps = 1e-9

for epoch in range(1, epochs + 1):
    hidden_input = X @ W1 + b1
    hidden_output = sigmoid(hidden_input)

    final_input = hidden_output @ W2 + b2
    predicted_output = sigmoid(final_input)

    loss = -np.mean(
        Y * np.log(predicted_output + eps) +
        (1.0 - Y) * np.log(1.0 - predicted_output + eps)
    )

    output_error = predicted_output - Y

    dW2 = hidden_output.T @ output_error / X.shape[0]
    db2 = np.mean(output_error, axis=0, keepdims=True)

    hidden_error = (output_error @ W2.T) * sigmoid_derivative(hidden_output)

    dW1 = X.T @ hidden_error / X.shape[0]
    db1 = np.mean(hidden_error, axis=0, keepdims=True)

    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1


hidden_output = sigmoid(X @ W1 + b1)
final_output = sigmoid(hidden_output @ W2 + b2)

classes = (final_output >= 0.5).astype(int)
accuracy = np.mean(classes == Y) * 100


print("Input data:")
print(X.astype(int))

print("\nExpected output:")
print(Y.astype(int))

print("\nNeural network output:")
print(np.round(final_output, 4))

print("\nPredicted classes:")
print(classes)

print(f"\nAccuracy: {accuracy:.2f}%")
print(f"Final loss: {loss:.6f}")

print("\nWeights W1:")
print(np.round(W1, 4))

print("\nBias b1:")
print(np.round(b1, 4))

print("\nWeights W2:")
print(np.round(W2, 4))

print("\nBias b2:")
print(np.round(b2, 4))