import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from model import MNISTModel,MNISTModel_2
from datetime import datetime
import os
from tqdm import tqdm
from torchsummary import summary

def train():
    # Set device
    device = torch.device('cpu')  # Force CPU
    
    # Load MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
    
    # Initialize model
    model = MNISTModel_2().to(device)
    
    # Display model summary
    print("\nModel Summary:")
    summary(model, (1, 28, 28))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train for 1 epoch
    model.train()
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for data, target in pbar:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
        acc = 100. * correct / total
        
        # Update progress bar
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 
                        'accuracy': f'{acc:.2f}%'})
    
    print(f'\nFinal Training Accuracy: {acc:.2f}%')
    
    # Save model with timestamp and accuracy
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = f'model_mnist_{timestamp}_acc{acc:.1f}.pth'
    torch.save(model.state_dict(), save_path)
    print(f'Model saved as {save_path}')
    
if __name__ == '__main__':
    train() 