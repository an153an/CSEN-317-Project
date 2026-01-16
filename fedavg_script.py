
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import copy
import os

from simple_nn import SimpleNN


# ============================================================================
# DATA SPLITTING
# ============================================================================

def split_data_non_iid(dataset, num_clients=5):
    """Split dataset into non-IID parts for each client"""
    labels = np.array(dataset.targets)
    num_classes = 10
    
    # Group indices by class
    class_indices = {i: np.where(labels == i)[0] for i in range(num_classes)}
    
    # Shuffle each class
    for indices in class_indices.values():
        np.random.shuffle(indices)
    
    # Distribute to clients (uneven random split)
    client_indices = [[] for _ in range(num_clients)]
    
    for class_id in range(num_classes):
        indices = class_indices[class_id]
        # Random uneven split using Dirichlet distribution
        proportions = np.random.dirichlet(np.ones(num_clients) * 0.5)
        proportions = (proportions * len(indices)).astype(int)
        proportions[-1] = len(indices) - proportions[:-1].sum()
        
        # Assign to clients
        start = 0
        for client_id, count in enumerate(proportions):
            client_indices[client_id].extend(indices[start:start + count])
            start += count
    
    # Print distribution
    print("\nData Distribution:")
    for i, indices in enumerate(client_indices):
        print(f"Client {i}: {len(indices)} samples")
    
    return [Subset(dataset, idx) for idx in client_indices], [len(idx) for idx in client_indices]


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_client(model, dataloader, device, epochs=5, lr=0.01):
    """Train model on client data"""
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    
    correct, total = 0, 0
    for _ in range(epochs):
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy


def evaluate_model(model, dataloader, device):
    """Evaluate model on test data"""
    model.eval()
    correct, total = 0, 0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return 100 * correct / total


def aggregate_models(models, client_sizes):
    """Average model weights (FedAvg)"""
    total_size = sum(client_sizes)
    global_dict = copy.deepcopy(models[0].state_dict())
    
    for key in global_dict.keys():
        global_dict[key] = torch.zeros_like(global_dict[key])
        for i, model in enumerate(models):
            global_dict[key] += model.state_dict()[key] * (client_sizes[i] / total_size)
    
    return global_dict


# ============================================================================
# MAIN FEDERATED TRAINING
# ============================================================================

def federated_training(num_rounds=50, num_clients=5, local_epochs=5):
    """Run one FedAvg experiment"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    train_dataset = datasets.FashionMNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST('./data', train=False, download=True, transform=transform)
    
    # Split data among clients
    client_datasets, client_sizes = split_data_non_iid(train_dataset, num_clients)
    client_loaders = [DataLoader(ds, batch_size=64, shuffle=True) for ds in client_datasets]
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Initialize models
    global_model = SimpleNN().to(device)
    client_models = [SimpleNN().to(device) for _ in range(num_clients)]
    
    # Track accuracies
    client_train_accs = [[] for _ in range(num_clients)]
    server_test_accs = []
    
    print(f"\nStarting FedAvg Training")
    print("="*60)
    
    for round_num in range(num_rounds):
        print(f"\nRound {round_num + 1}/{num_rounds}")
        
        # Send global model to clients
        global_weights = global_model.state_dict()
        for model in client_models:
            model.load_state_dict(copy.deepcopy(global_weights))
        
        # Each client trains locally
        for i, (model, loader) in enumerate(zip(client_models, client_loaders)):
            train_acc = train_client(model, loader, device, local_epochs)
            client_train_accs[i].append(train_acc)
            print(f"  Client {i}: {train_acc:.2f}%")
        
        # Server aggregates models
        aggregated_weights = aggregate_models(client_models, client_sizes)
        global_model.load_state_dict(aggregated_weights)
        
        # Evaluate global model
        test_acc = evaluate_model(global_model, test_loader, device)
        server_test_accs.append(test_acc)
        print(f"  Server Test: {test_acc:.2f}%")
    
    return client_train_accs, server_test_accs


# ============================================================================
# RUN EXPERIMENT
# ============================================================================

def run_experiment(num_repeats=5, num_rounds=50):
    """Run FedAvg multiple times"""
    all_client_accs = []
    all_server_accs = []
    final_accs = []
    
    for repeat in range(num_repeats):
        print(f"\n{'='*60}")
        print(f"REPEAT {repeat + 1}/{num_repeats}")
        print(f"{'='*60}")
        
        torch.manual_seed(42 + repeat)
        np.random.seed(42 + repeat)
        
        client_accs, server_accs = federated_training(num_rounds=num_rounds)
        
        all_client_accs.append(client_accs)
        all_server_accs.append(server_accs)
        final_accs.append(server_accs[-1])
    
    # Print results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Final Test Accuracy: {np.mean(final_accs):.2f}% ± {np.std(final_accs):.2f}%")
    print(f"Individual runs: {[f'{acc:.2f}%' for acc in final_accs]}")
    
    return all_client_accs, all_server_accs


# ============================================================================
# PLOTTING
# ============================================================================

def plot_results(all_client_accs, all_server_accs):
    """Plot training results"""
    os.makedirs('results', exist_ok=True)
    
    # Use first run for client plots
    client_accs = all_client_accs[0]
    
    # Average server accuracies across runs
    server_accs = np.array(all_server_accs)
    mean_server = np.mean(server_accs, axis=0)
    std_server = np.std(server_accs, axis=0)
    
    rounds = range(1, len(mean_server) + 1)
    
    # Create plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Client training accuracies
    colors = ['b', 'g', 'r', 'c', 'm']
    for i in range(5):
        ax1.plot(rounds, client_accs[i], color=colors[i], label=f'Client {i}', linewidth=2)
    ax1.set_xlabel('Communication Round', fontsize=12)
    ax1.set_ylabel('Training Accuracy (%)', fontsize=12)
    ax1.set_title('Client Training Accuracies', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Server test accuracy
    ax2.plot(rounds, mean_server, 'r-', label='Server Test', linewidth=2)
    ax2.fill_between(rounds, mean_server - std_server, mean_server + std_server, alpha=0.2, color='r')
    ax2.set_xlabel('Communication Round', fontsize=12)
    ax2.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax2.set_title('Server Test Accuracy (Avg of 5 runs)', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/fedavg_results.png', dpi=300)
    print("\nPlot saved to results/fedavg_results.png")
    plt.show()


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Run experiment
    all_client_accs, all_server_accs = run_experiment(num_repeats=5, num_rounds=50)
    
    # Plot results
    plot_results(all_client_accs, all_server_accs)
    
    print("\nDone!")



