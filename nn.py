import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

class DynamicCNN(nn.Module):
    def __init__(self, config):
        super(DynamicCNN, self).__init__()
        self.layers = nn.ModuleList()
        
        # Input layer
        in_channels = config['inputLayer']['channels']
        input_height = config['inputLayer']['height']
        input_width = config['inputLayer']['width']
        
        # Convolutional layers
        for conv, act, pool in zip(config['convLayers'], config['activationLayers'], config['poolingLayers']):
            self.layers.append(nn.Conv2d(in_channels=in_channels, out_channels=conv['filters'], 
                                         kernel_size=conv['kernelSize'], stride=conv['stride'], 
                                         padding=conv['padding']))
            self.layers.append(self.get_activation(act))
            self.layers.append(nn.MaxPool2d(kernel_size=pool['kernelSize'], stride=pool['stride']))
            in_channels = conv['filters']
        
        # Calculate the size of the flattened feature map
        with torch.no_grad():
            x = torch.randn(1, config['inputLayer']['channels'], input_height, input_width)
            for layer in self.layers:
                x = layer(x)
            flattened_size = x.view(1, -1).size(1)
        
        # Fully connected layers
        self.layers.append(nn.Flatten())
        prev_units = flattened_size
        for fc in config['FullyConnectedLayer']:
            self.layers.append(nn.Linear(prev_units, fc['units']))
            self.layers.append(self.get_activation(fc.get('activation', 'relu')))
            prev_units = fc['units']
        
        # Output layer
        self.layers.append(nn.Linear(config['FullyConnectedLayer'][-1]['units'], config['outputLayer']['units']))
        self.layers.append(self.get_activation(config['outputLayer']['activation']))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    @staticmethod
    def get_activation(name):
        if name.lower() == 'relu':
            return nn.ReLU()
        elif name.lower() == 'softmax':
            return nn.Softmax(dim=1)
        elif name.lower() == 'sigmoid':
            return nn.Sigmoid()
        else:
            raise ValueError(f"Unsupported activation function: {name}")

def create_model_from_config(config):
    model = DynamicCNN(config)
    
    # Set up optimizer
    optimizer_class = getattr(optim, config['optimizer'])
    optimizer = optimizer_class(model.parameters(), lr=config['modelConf']['learningRate'])
    
    # Set up loss function
    criterion = getattr(nn, config['lossFunction'])()
    
    return model, optimizer, criterion

def train(model, optimizer, criterion, train_loader, val_loader, epochs, device):
    model.to(device)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = 100. * correct / total
        
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print("--------------------")

def main():
    # Configuration
    config = {
        'inputLayer': {'channels': 1, 'height': 28, 'width': 28},
        'convLayers': [
            {'filters': 32, 'kernelSize': 3, 'stride': 1, 'padding': 1},
            {'filters': 64, 'kernelSize': 3, 'stride': 1, 'padding': 1}
        ],
        'activationLayers': ['relu', 'relu'],
        'poolingLayers': [
            {'kernelSize': 2, 'stride': 2},
            {'kernelSize': 2, 'stride': 2}
        ],
        'FullyConnectedLayer': [
            {'units': 128, 'activation': 'relu'},
            {'units': 64, 'activation': 'relu'}
        ],
        'outputLayer': {'units': 10, 'activation': 'softmax'},
        'optimizer': 'Adam',
        'lossFunction': 'CrossEntropyLoss',
        'modelConf': {'learningRate': 0.001, 'epochs': 10}
    }

    # Create model, optimizer, and criterion
    model, optimizer, criterion = create_model_from_config(config)

    # MNIST dataset and data loaders
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    val_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(model)
    print(f"Optimizer: {optimizer}")
    print(f"Loss function: {criterion}")
    print(f"Device: {device}")

    # Train the model
    train(model, optimizer, criterion, train_loader, val_loader, config['modelConf']['epochs'], device)

if __name__ == "__main__":
    main()