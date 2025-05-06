from __future__ import print_function

import argparse
import numpy as np
import torch
import torch.optim as optim
import torch.utils.data as data_utils
from torch.autograd import Variable
from torchmetrics import Accuracy, Precision, Recall, F1Score, AUROC

from dataloader import MnistBags
from model import Attention, GatedAttention

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST bags Example')
parser.add_argument('--epochs', type=int, default=20, metavar='N',
                   help='number of epochs to train (default: 20)')
parser.add_argument('--lr', type=float, default=0.0005, metavar='LR',
                   help='learning rate (default: 0.0005)')
parser.add_argument('--reg', type=float, default=10e-5, metavar='R',
                   help='weight decay')
parser.add_argument('--target_number', type=int, default=9, metavar='T',
                   help='bags have positive labels if they contain at least one 9')
parser.add_argument('--mean_bag_length', type=int, default=10, metavar='ML',
                   help='average bag length')
parser.add_argument('--var_bag_length', type=int, default=2, metavar='VL',
                   help='variance of bag length')
parser.add_argument('--num_bags_train', type=int, default=200, metavar='NTrain',
                   help='number of bags in training set')
parser.add_argument('--num_bags_test', type=int, default=50, metavar='NTest',
                   help='number of bags in test set')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                   help='random seed (default: 1)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                   help='disables CUDA training')
parser.add_argument('--model', type=str, default='attention', 
                   help='Choose b/w attention and gated_attention')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    print('\nGPU is ON!')

print('Load Train and Test Set')
loader_kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

train_loader = data_utils.DataLoader(MnistBags(target_number=args.target_number,
                                          mean_bag_length=args.mean_bag_length,
                                          var_bag_length=args.var_bag_length,
                                          num_bag=args.num_bags_train,
                                          seed=args.seed,
                                          train=True),
                                batch_size=1,
                                shuffle=True,
                                **loader_kwargs)

test_loader = data_utils.DataLoader(MnistBags(target_number=args.target_number,
                                         mean_bag_length=args.mean_bag_length,
                                         var_bag_length=args.var_bag_length,
                                         num_bag=args.num_bags_test,
                                         seed=args.seed,
                                         train=False),
                               batch_size=1,
                               shuffle=False,
                               **loader_kwargs)

print('Init Model')
if args.model == 'attention':
    model = Attention()
elif args.model == 'gated_attention':
    model = GatedAttention()
if args.cuda:
    model.cuda()

# Initialize metrics
def init_metrics(device):
    return {
        'accuracy': Accuracy(task='binary').to(device),
        'precision': Precision(task='binary').to(device),
        'recall': Recall(task='binary').to(device),
        'f1': F1Score(task='binary').to(device),
        'auroc': AUROC(task='binary').to(device)
    }

train_metrics = init_metrics('cuda' if args.cuda else 'cpu')
test_metrics = init_metrics('cuda' if args.cuda else 'cpu')

optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.reg)

def update_metrics(metrics, y_prob, y_true):
    y_pred = (y_prob >= 0.5).float()
    for metric in metrics.values():
        metric.update(y_pred, y_true)
    # AUROC needs probabilities
    metrics['auroc'].update(y_prob, y_true)

def log_metrics(metrics, prefix):
    results = {}
    for name, metric in metrics.items():
        results[name] = metric.compute()
        metric.reset()  # Reset after logging
    print(f"{prefix} - " + ", ".join([f"{k}: {v:.4f}" for k, v in results.items()]))
    return results

def train(epoch):
    model.train()
    train_loss = 0.
    
    for batch_idx, (data, label) in enumerate(train_loader):
        bag_label = label[0].float()  # Convert to float tensor
        if args.cuda:
            data, bag_label = data.cuda(), bag_label.cuda()
        data, bag_label = Variable(data), Variable(bag_label)

        # reset gradients
        optimizer.zero_grad()
        # forward pass
        y_prob, y_hat, _ = model(data)
        # calculate loss - ensure all tensors are float
        y_prob = y_prob.float()
        bag_label = bag_label.float()
        loss = -1. * (bag_label * torch.log(y_prob + 1e-10) + (1. - bag_label) * torch.log(1. - y_prob + 1e-10))
        train_loss += loss.item()
        # backward pass
        loss.backward()
        # step
        optimizer.step()
        
        # Update metrics
        update_metrics(train_metrics, y_prob, bag_label)

    # calculate average loss for epoch
    train_loss /= len(train_loader)
    
    # Log metrics
    print(f'Epoch: {epoch}, Loss: {train_loss:.4f}')
    log_metrics(train_metrics, prefix='Train')

def test():
    model.eval()
    test_loss = 0.
    
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(test_loader):
            bag_label = label[0].float()  # Convert to float tensor
            instance_labels = label[1]
            if args.cuda:
                data, bag_label = data.cuda(), bag_label.cuda()
            data, bag_label = Variable(data), Variable(bag_label)
            
            y_prob, y_hat, attention_weights = model(data)
            # calculate loss - ensure all tensors are float
            y_prob = y_prob.float()
            bag_label = bag_label.float()
            loss = -1. * (bag_label * torch.log(y_prob + 1e-10) + (1. - bag_label) * torch.log(1. - y_prob + 1e-10))
            test_loss += loss.item()
            
            # Update metrics
            update_metrics(test_metrics, y_prob, bag_label)

            if batch_idx < 5:  # plot bag labels and instance labels for first 5 bags
                bag_level = (bag_label.cpu().data.numpy()[0], int(y_hat.cpu().data.numpy()[0][0]))
                instance_level = list(zip(instance_labels.numpy()[0].tolist(),
                                    np.round(attention_weights.cpu().data.numpy()[0], decimals=3).tolist()))

                print('\nTrue Bag Label, Predicted Bag Label: {}\n'
                     'True Instance Labels, Attention Weights: {}'.format(bag_level, instance_level))

    # calculate average loss
    test_loss /= len(test_loader)
    
    # Log metrics
    print(f'\nTest Set - Loss: {test_loss:.4f}')
    test_results = log_metrics(test_metrics, prefix='Test')
    return test_results

if __name__ == "__main__":
    print('Start Training')
    best_f1 = 0
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test_results = test()
        
        # Track best model based on F1 score
        if test_results['f1'] > best_f1:
            best_f1 = test_results['f1']
            torch.save(model.state_dict(), 'best_model.pth')
            print(f'New best model saved with F1: {best_f1:.4f}')
    
    print('Training Complete')
    print(f'Best Test F1: {best_f1:.4f}')