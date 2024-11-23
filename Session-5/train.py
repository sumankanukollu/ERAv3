import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from model import selected_model  # Import the global instance
from datetime import datetime
import os
from tqdm import tqdm
from torchsummary import summary
import torchvision.utils as vutils
import numpy as np
import random
import sys

def set_seed(seed=42):
    """Set all seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def save_augmented_samples(dataset, num_images=100, save_dir='augmented_samples'):
    # Create directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Get a dataloader with batch size 1
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    
    # Save specified number of images
    for i, (img, label) in enumerate(loader):
        if i >= num_images:
            break
        
        # Save the image
        vutils.save_image(
            img,
            os.path.join(save_dir, f'augmented_sample_{i}_label_{label.item()}.png'),
            normalize=True
        )

def train():
    # Set seed for reproducibility
    set_seed(42)
    
    # Set device
    device = torch.device('cpu')  # Force CPU
    
    # Move global model to device
    model = selected_model.to(device)
    
    # Load MNIST dataset with augmentation
    train_transform = transforms.Compose([
        transforms.RandomRotation(15),  # Random rotation up to 15 degrees
        transforms.RandomAffine(0, translate=(0.1, 0.1)),  # Random shift up to 10%
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.RandomErasing(p=0.2)  # Randomly erase parts of image
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
    
    # Save augmented samples
    print("Saving augmented samples...")
    save_augmented_samples(train_dataset, num_images=100)
    print("Augmented samples saved in 'augmented_samples' directory")
    
    # Display model summary
    print("\nModel Summary:")
    summary(model, (1, 28, 28))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train for 1 epoch
    model.train()
    correct = 0
    total = 0
    
    # Modify tqdm to be less verbose
    pbar = tqdm(train_loader, 
                desc='Training',
                ncols=100,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
                file=sys.stdout)
    
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
        
        # Update progress bar less frequently (every 50 batches)
        if total % (50 * target.size(0)) == 0:
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}', 
                'accuracy': f'{acc:.2f}%'
            }, refresh=True)
    
    print(f'\nFinal Training Accuracy: {acc:.2f}%')
    
    # Save model with timestamp and accuracy
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = f'model_mnist_{timestamp}_acc{acc:.1f}.pth'
    torch.save(model.state_dict(), save_path)
    print(f'Model saved as {save_path}')
    
if __name__ == '__main__':
    train() 