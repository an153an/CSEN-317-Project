
import torch
import torch.nn as nn

class SimpleNN(nn.Module):
    """
    Simple 2-layer neural network for image classification
    
    Architecture:
    - Input layer: flattened image (28x28 = 784 for FashionMNIST)
    - Hidden layer: fully connected with ReLU activation
    - Output layer: fully connected (10 classes)
    """
    
    def __init__(self, input_size=784, hidden_size=128, num_classes=10):
        """
        Initialize the neural network
        
        Args:
            input_size: Size of input features (784 for 28x28 images)
            hidden_size: Number of neurons in hidden layer
            num_classes: Number of output classes (10 for FashionMNIST/CIFAR-10)
        """
        super(SimpleNN, self).__init__()
        
        # First layer: input -> hidden
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        
        # Second layer: hidden -> output
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
        
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        # Flatten the input: (batch_size, C, H, W) -> (batch_size, C*H*W)
        x = x.view(x.size(0), -1)
        
        # First layer with activation
        x = self.fc1(x)
        x = self.relu(x)
        
        # Second layer (output)
        x = self.fc2(x)
        
        return x
    
    def get_weights(self):
        """
        Get model weights as a dictionary (for FedAvg)
        
        Returns:
            Dictionary of model state
        """
        return self.state_dict()
    
    def set_weights(self, weights):
        """
        Set model weights from a dictionary (for FedAvg)
        
        Args:
            weights: Dictionary of model state
        """
        self.load_state_dict(weights)



