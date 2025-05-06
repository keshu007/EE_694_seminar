import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchmetrics import Accuracy, Precision, Recall, F1Score, AUROC
from medmnist import PathMNIST

class PathMNISTBags(Dataset):
    """Modified PathMNIST dataset organized in bags for MIL"""
    def __init__(self, target_class=0, mean_bag_length=10, var_bag_length=2, 
                 num_bag=200, seed=1, train=True):
        self.target_class = target_class
        self.mean_bag_length = mean_bag_length
        self.var_bag_length = var_bag_length
        self.num_bag = num_bag
        self.train = train
        
        # Load original PathMNIST dataset
        self.pathmnist = PathMNIST(root='./data', split='train' if train else 'test',
                                  transform=transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                  ]), download=True)
        
        # Set random seed for reproducibility
        self.r = np.random.RandomState(seed)
        
        # Create bags
        self.bags_list, self.labels_list = self._create_bags()
        
    def _create_bags(self):
        bags = []
        labels = []
        
        for _ in range(self.num_bag):
            # Randomly determine bag length
            bag_length = max(1, int(self.r.normal(self.mean_bag_length, self.var_bag_length)))
            
            # Randomly select instances
            indices = torch.LongTensor(self.r.choice(len(self.pathmnist), bag_length, replace=False))
            instances = torch.stack([self.pathmnist[i][0] for i in indices])
            instance_labels = torch.tensor([self.pathmnist[i][1] for i in indices])
            
            # Create bag label (1 if at least one instance is target_class)
            bag_label = 1 if (instance_labels == self.target_class).any() else 0
            
            # Store bag and label
            bags.append(instances)  # Shape: [bag_length, 3, 28, 28]
            labels.append([bag_label, instance_labels])
            
        return bags, labels
    
    def __len__(self):
        return len(self.labels_list)
    
    def __getitem__(self, index):
        return self.bags_list[index], self.labels_list[index]

class AttentionMIL(nn.Module):
    """Attention-based MIL model for PathMNIST"""
    def __init__(self):
        super(AttentionMIL, self).__init__()
        self.L = 128  # Attention dimension
        self.D = 128  # Hidden dimension
        self.K = 1    # Number of attention branches
        
        # Feature extractor (CNN for PathMNIST - 3 channel input)
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 20, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(50 * 4 * 4, self.L),
            nn.Tanh(),
            nn.Linear(self.L, self.K)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(50 * 4 * 4 * self.K, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # x shape: (batch, instances, channels, height, width)
        batch_size, num_instances = x.size(0), x.size(1)
        
        # Reshape for feature extraction
        x = x.view(-1, *x.shape[2:])  # (batch*instances, channels, height, width)
        
        # Extract features
        H = self.feature_extractor(x)  # (batch*instances, 50, 4, 4)
        H = H.view(batch_size, num_instances, -1)  # (batch, instances, 50*4*4)
        
        # Attention weights
        A = self.attention(H)  # (batch, instances, K)
        A = torch.transpose(A, 2, 1)  # (batch, K, instances)
        A = F.softmax(A, dim=2)  # Softmax over instances
        
        # Apply attention
        Z = torch.bmm(A, H)  # (batch, K, 50*4*4)
        Z = Z.view(batch_size, -1)  # (batch, K*50*4*4)
        
        # Classification
        Y_prob = self.classifier(Z)
        Y_hat = torch.ge(Y_prob, 0.5).float()
        
        return Y_prob, Y_hat, A

def init_metrics(device):
    """Initialize metrics for evaluation"""
    return {
        'accuracy': Accuracy(task='binary').to(device),
        'precision': Precision(task='binary').to(device),
        'recall': Recall(task='binary').to(device),
        'f1': F1Score(task='binary').to(device),
        'auroc': AUROC(task='binary').to(device)
    }

def train(model, train_loader, optimizer, metrics, device, epoch):
    """Training loop"""
    model.train()
    train_loss = 0.
    
    for batch_idx, (data, label) in enumerate(train_loader):
        bag_label = label[0].float().to(device)
        data = data.to(device)
        
        # Reset gradients
        optimizer.zero_grad()
        
        # Forward pass
        y_prob, y_hat, _ = model(data)
        
        # Calculate loss
        loss = -1. * (bag_label * torch.log(y_prob + 1e-10) + 
                     (1. - bag_label) * torch.log(1. - y_prob + 1e-10))
        train_loss += loss.item()
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Update metrics - reshape tensors to match expected shapes
        y_hat = y_hat.view(-1)  # Reshape to [batch_size]
        bag_label = bag_label.view(-1)  # Reshape to [batch_size]
        
        for metric in metrics.values():
            if isinstance(metric, AUROC):
                y_prob = y_prob.view(-1)  # For AUROC we use probabilities
                metric.update(y_prob, bag_label)
            else:
                metric.update(y_hat, bag_label)
    
    # Calculate average loss
    train_loss /= len(train_loader)
    
    # Compute and log metrics
    results = {}
    for name, metric in metrics.items():
        results[name] = metric.compute()
        metric.reset()
    
    print(f'Epoch: {epoch}, Loss: {train_loss:.4f}')
    print(f"Train - " + ", ".join([f"{k}: {v:.4f}" for k, v in results.items()]))
    
    return results

def test(model, test_loader, metrics, device):
    """Testing loop"""
    model.eval()
    test_loss = 0.
    
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(test_loader):
            bag_label = label[0].float().to(device)
            data = data.to(device)
            
            # Forward pass
            y_prob, y_hat, attention_weights = model(data)
            
            # Calculate loss
            loss = -1. * (bag_label * torch.log(y_prob + 1e-10) + 
                         (1. - bag_label) * torch.log(1. - y_prob + 1e-10))
            test_loss += loss.item()
            
            # Update metrics - reshape tensors to match expected shapes
            y_hat = y_hat.view(-1)  # Reshape to [batch_size]
            bag_label = bag_label.view(-1)  # Reshape to [batch_size]
            
            for metric in metrics.values():
                if isinstance(metric, AUROC):
                    y_prob = y_prob.view(-1)  # For AUROC we use probabilities
                    metric.update(y_prob, bag_label)
                else:
                    metric.update(y_hat, bag_label)
            
            # Print first 5 bags for inspection
            if batch_idx < 5:
                print(f"\nBag {batch_idx}:")
                print(f"True Label: {bag_label.item():.0f}, Predicted: {y_hat.item():.0f}")
                print("Attention weights:", attention_weights.cpu().numpy()[0, 0])
    
    # Calculate average loss
    test_loss /= len(test_loader)
    
    # Compute and log metrics
    results = {}
    for name, metric in metrics.items():
        results[name] = metric.compute()
        metric.reset()
    
    print(f'\nTest - Loss: {test_loss:.4f}')
    print(f"Test - " + ", ".join([f"{k}: {v:.4f}" for k, v in results.items()]))
    
    return results

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch PathMNIST MIL Example')
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
    parser.add_argument('--reg', type=float, default=10e-5, help='weight decay')
    parser.add_argument('--target_class', type=int, default=0, 
                       help='positive class (0-8 for PathMNIST)')
    parser.add_argument('--mean_bag_length', type=int, default=10, help='avg bag size')
    parser.add_argument('--var_bag_length', type=int, default=2, help='bag size variance')
    parser.add_argument('--num_bags_train', type=int, default=200, help='train bags')
    parser.add_argument('--num_bags_test', type=int, default=50, help='test bags')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disable CUDA')
    args = parser.parse_args()
    
    # Device setup
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.manual_seed(args.seed)
    
    # Create datasets
    train_dataset = PathMNISTBags(
        target_class=args.target_class,
        mean_bag_length=args.mean_bag_length,
        var_bag_length=args.var_bag_length,
        num_bag=args.num_bags_train,
        seed=args.seed,
        train=True
    )
    
    test_dataset = PathMNISTBags(
        target_class=args.target_class,
        mean_bag_length=args.mean_bag_length,
        var_bag_length=args.var_bag_length,
        num_bag=args.num_bags_test,
        seed=args.seed,
        train=False
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # Initialize model and optimizer
    model = AttentionMIL().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.reg)
    
    # Initialize metrics
    train_metrics = init_metrics(device)
    test_metrics = init_metrics(device)
    
    # Training loop
    best_f1 = 0
    for epoch in range(1, args.epochs + 1):
        train_results = train(model, train_loader, optimizer, train_metrics, device, epoch)
        test_results = test(model, test_loader, test_metrics, device)
        
        # Save best model based on F1 score
        if test_results['f1'] > best_f1:
            best_f1 = test_results['f1']
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"Saved new best model with F1: {best_f1:.4f}")
    
    print(f"\nTraining complete. Best test F1: {best_f1:.4f}")

if __name__ == '__main__':
    main()
