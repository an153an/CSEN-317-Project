
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import os

from simple_nn import SimpleNN

def load_fashion_mnist(batch_size=64):
    """
    Load FashionMNIST dataset
    Returns: train_loader, test_loader
    """

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Training Data
    train_dataset = datasets.FashionMNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )

    # Testing Data
    test_dataset = datasets.FashionMNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Testing samples: {len(test_dataset)}")
    
    return train_loader, test_loader

def create_model(input_size=784, hidden_size=128, num_classes=10):
    """
    Create a simple 2-layer neural network
    Returns: model
    """

    model = SimpleNN(input_size, hidden_size, num_classes)
    return model

def train_baseline(model, train_loader, test_loader, epochs=20, lr=0.01):
    """
    Train the baseline model on full dataset
    Returns: train_accuracies, test_accuracies
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    train_accuracies = []
    test_accuracies = []

    for epoch in range(epochs):
        # Training phase
        model.train()
        correct = 0
        total = 0
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss += loss.item()
        
        train_acc = 100 * correct / total
        train_accuracies.append(train_acc)
        
        # Testing phase
        test_acc = evaluate_model(model, test_loader, device)
        test_accuracies.append(test_acc)
        
        print(f'Epoch [{epoch+1}/{epochs}], '
              f'Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%')
    
    return train_accuracies, test_accuracies

def evaluate_model(model, test_loader, device):
    """
    Evaluate model accuracy on test set
    Returns: accuracy (%)
    """

    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy

def run_experiment(num_repeats=5, epochs=20):
    """
    Run baseline training multiple times
    Returns: all_train_accs, all_test_accs, final_accs
    """

    all_train_accs = []
    all_test_accs = []
    final_test_accs = []
    
    print("="*60)
    print("RUNNING BASELINE EXPERIMENT")
    print("="*60)
    
    # Load data once
    train_loader, test_loader = load_fashion_mnist()
    
    for repeat in range(num_repeats):
        print(f"\n--- Repeat {repeat+1}/{num_repeats} ---")
        
        # Create fresh model for each repeat
        model = create_model()
        
        # Train model
        train_accs, test_accs = train_baseline(model, train_loader, test_loader, epochs)
        
        all_train_accs.append(train_accs)
        all_test_accs.append(test_accs)
        final_test_accs.append(test_accs[-1])
    
    # Calculate statistics
    mean_final_acc = np.mean(final_test_accs)
    std_final_acc = np.std(final_test_accs)
    
    print("\n" + "="*60)
    print("BASELINE RESULTS")
    print("="*60)
    print(f"Final Test Accuracy: {mean_final_acc:.2f}% ± {std_final_acc:.2f}%")
    print(f"Individual runs: {[f'{acc:.2f}%' for acc in final_test_accs]}")
    
    return all_train_accs, all_test_accs, final_test_accs


def plot_results(all_train_accs, all_test_accs, save_path='results/baseline_results.png'):
    """
    Plot training and testing accuracy curves
    """

    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Convert to numpy arrays
    all_train_accs = np.array(all_train_accs)
    all_test_accs = np.array(all_test_accs)
    
    # Calculate mean and std
    mean_train = np.mean(all_train_accs, axis=0)
    std_train = np.std(all_train_accs, axis=0)
    mean_test = np.mean(all_test_accs, axis=0)
    std_test = np.std(all_test_accs, axis=0)
    
    epochs = range(1, len(mean_train) + 1)
    
    # Create plot
    plt.figure(figsize=(12, 6))
    
    # Plot training accuracy
    plt.plot(epochs, mean_train, 'b-', label='Training Accuracy', linewidth=2)
    plt.fill_between(epochs, mean_train - std_train, mean_train + std_train, 
                     alpha=0.2, color='b')
    
    # Plot testing accuracy
    plt.plot(epochs, mean_test, 'r-', label='Testing Accuracy', linewidth=2)
    plt.fill_between(epochs, mean_test - std_test, mean_test + std_test, 
                     alpha=0.2, color='r')
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Baseline Training: FashionMNIST (Average of 5 runs)', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to {save_path}")
    plt.show()

if __name__ == "__main__":

    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Run the experiment
    all_train_accs, all_test_accs, final_accs = run_experiment(num_repeats=5, epochs=20)
    
    # Plot results
    plot_results(all_train_accs, all_test_accs)
    
    print("\nBaseline experiment completed!")


