#Part 1: Set up and data preparation

# Import necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import time
import os

# Apply transformations to normalize the data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # Normalize with mean and std for MNIST
])

# Load the MNIST dataset
mnist_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)

# Extract data and targets from the dataset
X = mnist_dataset.data  # Shape will be [60000, 28, 28] for images
y = mnist_dataset.targets  # Shape will be [60000] for labels

# Flatten the images to be of shape [60000, 784]
X = X.view(X.size(0), -1).float()  # Reshape to [60000, 784]

# Split the dataset into training and test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Part 2: Model Building
# Defining a logistic regression model using PyTorch
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

# Setting input and output dimensions
input_dim = 28 * 28  # MNIST images are 28x28
output_dim = 10      # There are 10 classes (digits 0-9)

# Creating the logistic regression model
model = LogisticRegressionModel(input_dim, output_dim)
model

# Defining loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Part 3: Training the Model
# Training the logistic regression model
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    # Forward pass
    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    # Backward pass and optimization
    loss.backward()
    optimizer.step()

    # Print loss every epoch
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluating the trained model
model.eval()
with torch.no_grad():
    outputs = model(X_test)
    _, predicted = torch.max(outputs, 1)
    test_accuracy = accuracy_score(y_test, predicted)

print(f'Test Accuracy: {test_accuracy * 100:.2f}%')

# Measuring the size of the original model
torch.save(model.state_dict(), "original_model.pth")
original_model_size = os.path.getsize("original_model.pth")
print(f"Original Model Size: {original_model_size / 1024:.2f} KB")

# Part 4: Quantization Function
def quantize_model(model, scale_factor=2 ** 7):
    # Copy the model to avoid modifying the original
    quantized_model = LogisticRegressionModel(input_dim, output_dim)
    quantized_model.load_state_dict(model.state_dict())

    # Quantize the weights to 8-bit
    with torch.no_grad():
        for param in quantized_model.parameters():
            param.data = torch.round(param.data * scale_factor) / scale_factor
    return quantized_model

# Part 5: Inference Function for the Quantized Model
def inference(model, X):
    model.eval()
    with torch.no_grad():
        outputs = model(X)
        _, predicted = torch.max(outputs, 1)
    return predicted

# Quantizing the model
quantized_model = quantize_model(model)
quantize_model

# Saving the quantized model size
torch.save(quantized_model.state_dict(), "quantized_model.pth")
quantized_model_size = os.path.getsize("quantized_model.pth")
print(f"Quantized Model Size: {quantized_model_size / 1024:.2f} KB")

# Measuring inference time for original and quantized models
def measure_inference_time(model, X):
    start_time = time.time()
    _ = inference(model, X)
    end_time = time.time()
    return (end_time - start_time) * 1000  # Convert to milliseconds

original_inference_time = measure_inference_time(model, X_test)
original_inference_time

quantized_inference_time = measure_inference_time(quantized_model, X_test)
quantized_inference_time

# Evaluating the quantized model
quantized_preds = inference(quantized_model, X_test)
quantized_preds

quantized_test_accuracy = accuracy_score(y_test, quantized_preds)
quantized_test_accuracy

# Part 6: Comparison Results
print(f"Quantized Test Accuracy: {quantized_test_accuracy * 100:.2f}%")
print(f"Original Inference Time: {original_inference_time:.2f} ms")
print(f"Quantized Inference Time: {quantized_inference_time:.2f} ms")
print(f"Original Model Size: {original_model_size / 1024:.2f} KB")
print(f"Quantized Model Size: {quantized_model_size / 1024:.2f} KB")
print(f"Original Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"Quantized Test Accuracy: {quantized_test_accuracy * 100:.2f}%")