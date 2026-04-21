import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import os
import sys

# Check dependencies
try:
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
except Exception as e:
    print(f"Error: Cannot import PyTorch - {e}")
    print("Please install dependencies: pip install -r requirements.txt")
    sys.exit(1)

# Create data directory
data_dir = './data'
if not os.path.exists(data_dir):
    try:
        os.makedirs(data_dir)
        print(f"Created data directory: {data_dir}")
    except Exception as e:
        print(f"Error: Cannot create data directory - {e}")
        sys.exit(1)
else:
    print(f"Data directory exists: {data_dir}")

if not os.access(data_dir, os.W_OK):
    print(f"Error: Data directory {data_dir} is not writable")
    sys.exit(1)

print("\nInitializing training environment...\n")
# Data augmentation and transformation
train_transform = transforms.Compose([
    transforms.RandomRotation(degrees=10),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Load MNIST dataset
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=train_transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=test_transform)

# Define data loaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)
# Define CNN model architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.25)
        self.dropout3 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.dropout1(x)
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.dropout2(x)
        x = torch.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(-1, 128 * 3 * 3)
        x = torch.relu(self.fc1(x))
        x = self.dropout3(x)
        x = self.fc2(x)
        output = torch.log_softmax(x, dim=1)
        return output

# Initialize model, loss function and optimizer
model = Net()
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

def train(model, device, train_loader, optimizer, epoch):
    """Training function"""
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

def test(model, device, test_loader):
    """Testing function"""
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')
    return accuracy

# Training configuration
device = torch.device("cpu")
model.to(device)

print(f"Starting training on {device}...")

# Train for 15 epochs
best_accuracy = 0.0
for epoch in range(1, 16):
    train(model, device, train_loader, optimizer, epoch)
    accuracy = test(model, device, test_loader)
    
    # Save best model
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        os.makedirs('./data', exist_ok=True)
        model_path = './data/digit_recognition_cnn.pth'
        torch.save(model.state_dict(), model_path)
        print(f"Best model saved to {model_path} (Accuracy: {accuracy:.2f}%)")

print(f"\nTraining completed! Best accuracy: {best_accuracy:.2f}%")
print(f"Final model saved to {model_path}")