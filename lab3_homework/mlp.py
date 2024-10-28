import numpy as np
from torchvision.datasets import MNIST
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split


# 1. Load and Preprocess the Data
def load_mnist(is_train: bool):
    dataset = MNIST(root='./data', transform=lambda x: np.array(x).flatten() / 255.0, download=True, train=is_train)
    data, labels = [], []
    for image, label in dataset:
        data.append(image)
        labels.append(label)
    return np.array(data), np.array(labels)


train_X, train_Y = load_mnist(True)
test_X, test_Y = load_mnist(False)

# 2. One-hot encode labels
encoder = OneHotEncoder(sparse_output=False)
train_Y = encoder.fit_transform(train_Y.reshape(-1, 1))
test_Y = encoder.transform(test_Y.reshape(-1, 1))

# Split into train and validation sets
train_X, val_X, train_Y, val_Y = train_test_split(train_X, train_Y, test_size=0.1, random_state=42)


# 3. MLP Model Definition
class MLP:
    def __init__(self, input_size=784, hidden_size=100, output_size=10, learning_rate=0.005, dropout_rate=0.2):
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate

        # Initialize weights and biases
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2. / input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2. / hidden_size)
        self.b2 = np.zeros((1, output_size))

    def relu(self, z):
        return np.maximum(0, z)

    def relu_derivative(self, z):
        return (z > 0).astype(float)

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    # 1. Forward Propagation
    def forward(self, X, is_training=True):
        # Layer 1
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.relu(self.z1)

        # Dropout on hidden layer during training
        if is_training:
            self.dropout_mask = (np.random.rand(*self.a1.shape) > self.dropout_rate).astype(np.float32)
            self.a1 *= self.dropout_mask
        else:
            self.a1 *= (1 - self.dropout_rate)

        # Output Layer
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.softmax(self.z2)
        return self.a2

    # Cross-entropy loss
    def compute_loss(self, y_pred, y_true):
        m = y_true.shape[0]
        log_likelihood = -np.log(y_pred[range(m), np.argmax(y_true, axis=1)])
        return np.sum(log_likelihood) / m

    # 2. Backpropagation
    def backprop(self, X, y_true, y_pred):
        m = X.shape[0]

        # Output layer gradients
        dz2 = y_pred - y_true
        dW2 = np.dot(self.a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m

        # Hidden layer gradients with dropout
        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * self.relu_derivative(self.z1)
        dz1 *= self.dropout_mask  # Apply dropout mask during backprop
        dW1 = np.dot(X.T, dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m

        # Parameter updates
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2

    # 3. Training the model with batch processing
    def train(self, X, y, epochs=10, batch_size=64):
        num_batches = X.shape[0] // batch_size
        for epoch in range(epochs):
            shuffled_indices = np.random.permutation(X.shape[0])
            X_shuffled = X[shuffled_indices]
            y_shuffled = y[shuffled_indices]

            epoch_loss = 0
            for i in range(num_batches):
                X_batch = X_shuffled[i * batch_size:(i + 1) * batch_size]
                y_batch = y_shuffled[i * batch_size:(i + 1) * batch_size]

                y_pred = self.forward(X_batch)
                loss = self.compute_loss(y_pred, y_batch)
                epoch_loss += loss

                self.backprop(X_batch, y_batch, y_pred)

            avg_loss = epoch_loss / num_batches
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}')

    # Prediction and Accuracy Calculation
    def predict(self, X):
        y_pred = self.forward(X, is_training=False)
        return np.argmax(y_pred, axis=1)

    def accuracy(self, y_pred, y_true):
        return np.mean(y_pred == y_true)


# 4. Training and Evaluation
mlp = MLP(learning_rate=0.005, dropout_rate=0.2)
epochs = 100
batch_size = 64

mlp.train(train_X, train_Y, epochs=epochs, batch_size=batch_size)

train_preds = mlp.predict(train_X)
train_acc = mlp.accuracy(train_preds, np.argmax(train_Y, axis=1))
print(f'Training Accuracy: {train_acc * 100:.2f}%')

val_preds = mlp.predict(val_X)
val_acc = mlp.accuracy(val_preds, np.argmax(val_Y, axis=1))
print(f'Validation Accuracy: {val_acc * 100:.2f}%')

test_preds = mlp.predict(test_X)
test_acc = mlp.accuracy(test_preds, np.argmax(test_Y, axis=1))
print(f'Test Accuracy: {test_acc * 100:.2f}%')
