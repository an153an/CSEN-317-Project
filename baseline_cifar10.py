
"""
CIFAR-10 Baseline Training
Same as baseline.py but using CIFAR-10 dataset
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import os

from simple_nn import SimpleNN


def load_cifar10(batch_size=64):
    """Load CIFAR-10 dataset"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10('./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Testing samples: {len(test_dataset)}")
    
    return train_loader, test_loader


def train_baseline(model, train_loader, test_loader, epochs=20, lr=0.01):
    """Train the baseline model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    
    train_accs, test_accs = [], []
    
    for epoch in range(epochs):
        model.train()
        correct, total = 0, 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_acc = 100 * correct / total
        train_accs.append(train_acc)
        
        # Evaluate
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        test_acc = 100 * correct / total
        test_accs.append(test_acc)
        
        print(f'Epoch [{epoch+1}/{epochs}], Train: {train_acc:.2f}%, Test: {test_acc:.2f}%')
    
    return train_accs, test_accs


def run_experiment(num_repeats=5, epochs=20):
    """Run baseline training multiple times"""
    all_train_accs, all_test_accs, final_accs = [], [], []
    
    print("="*60)
    print("CIFAR-10 BASELINE EXPERIMENT")
    print("="*60)
    
    train_loader, test_loader = load_cifar10()
    
    for repeat in range(num_repeats):
        print(f"\n--- Repeat {repeat+1}/{num_repeats} ---")
        model = SimpleNN(input_size=3072, hidden_size=256)
        train_accs, test_accs = train_baseline(model, train_loader, test_loader, epochs)
        
        all_train_accs.append(train_accs)
        all_test_accs.append(test_accs)
        final_accs.append(test_accs[-1])
    
    print("\n" + "="*60)
    print(f"Final Accuracy: {np.mean(final_accs):.2f}% ± {np.std(final_accs):.2f}%")
    print(f"Individual runs: {[f'{acc:.2f}%' for acc in final_accs]}")
    
    return all_train_accs, all_test_accs


def plot_results(all_train_accs, all_test_accs):
    """Plot results"""
    os.makedirs('results', exist_ok=True)
    
    all_train_accs = np.array(all_train_accs)
    all_test_accs = np.array(all_test_accs)
    
    mean_train = np.mean(all_train_accs, axis=0)
    std_train = np.std(all_train_accs, axis=0)
    mean_test = np.mean(all_test_accs, axis=0)
    std_test = np.std(all_test_accs, axis=0)
    
    epochs = range(1, len(mean_train) + 1)
    
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, mean_train, 'b-', label='Training', linewidth=2)
    plt.fill_between(epochs, mean_train - std_train, mean_train + std_train, alpha=0.2, color='b')
    plt.plot(epochs, mean_test, 'r-', label='Testing', linewidth=2)
    plt.fill_between(epochs, mean_test - std_test, mean_test + std_test, alpha=0.2, color='r')
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('CIFAR-10 Baseline (Avg of 5 runs)', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/baseline_cifar10.png', dpi=300)
    print("\nPlot saved to results/baseline_cifar10.png")
    plt.show()


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    
    all_train_accs, all_test_accs = run_experiment(num_repeats=5, epochs=20)
    plot_results(all_train_accs, all_test_accs)
    
    print("\nDone!")


