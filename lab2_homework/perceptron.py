import numpy as np
from torchvision.datasets import MNIST
from sklearn.preprocessing import OneHotEncoder

def download_mnist(is_train: bool):
    dataset = MNIST(root='./data', transform=lambda x: np.array(x).flatten() / 255.0,  # Normalize the data
                    download=True, train=is_train)
    mnist_data, mnist_labels = [], []
    for image, label in dataset:
        mnist_data.append(image)
        mnist_labels.append(label)
    return np.array(mnist_data), np.array(mnist_labels)


train_X, train_Y = download_mnist(True)
test_X, test_Y = download_mnist(False)

#  2.Normalize the data
encoder = OneHotEncoder(sparse_output=False)
train_Y_onehot = encoder.fit_transform(train_Y.reshape(-1, 1))
test_Y_onehot = encoder.transform(test_Y.reshape(-1, 1))

# 3. Training the perceptron
input_size = 784
num_classes = 10
np.random.seed(42)
W = np.random.randn(input_size, num_classes) * 0.01
b = np.zeros((num_classes,))


def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def forward_propagation(X, W, b):
    z = np.dot(X, W) + b
    return softmax(z)


def cross_entropy_loss(y_pred, y_true):
    m = y_true.shape[0]
    log_likelihood = -np.log(y_pred[range(m), np.argmax(y_true, axis=1)])
    loss = np.sum(log_likelihood) / m
    return loss


def backward_propagation(X, y_true, y_pred, W, b, learning_rate):
    m = X.shape[0]

    dz = y_pred - y_true
    dw = np.dot(X.T, dz) / m
    db = np.sum(dz, axis=0) / m

    W -= learning_rate * dw
    b -= learning_rate * db

    return W, b

def train_perceptron(train_X, train_Y, W, b, epochs, batch_size, learning_rate):
    num_batches = train_X.shape[0] // batch_size
    for epoch in range(epochs):
        epoch_loss = 0
        for i in range(num_batches):
            # Get the next batch
            batch_X = train_X[i * batch_size:(i + 1) * batch_size]
            batch_Y = train_Y[i * batch_size:(i + 1) * batch_size]

            # Forward propagation
            y_pred = forward_propagation(batch_X, W, b)

            # Compute loss
            loss = cross_entropy_loss(y_pred, batch_Y)
            epoch_loss += loss

            # Backward propagation and update weights
            W, b = backward_propagation(batch_X, batch_Y, y_pred, W, b, learning_rate)

        print(f'Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / num_batches}')
    return W, b


def predict(X, W, b):
    y_pred = forward_propagation(X, W, b)
    return np.argmax(y_pred, axis=1)


def accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)


epochs = 50  # range from 50-500
batch_size = 100
learning_rate = 0.1

W, b = train_perceptron(train_X, train_Y_onehot, W, b, epochs, batch_size, learning_rate)

y_test_pred = predict(test_X, W, b)
test_acc = accuracy(y_test_pred, test_Y)
print(f'Test Accuracy: {test_acc * 100:.2f}%')
