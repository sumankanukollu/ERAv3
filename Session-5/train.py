from matplotlib import pyplot as plt
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

# Visualize and save randomly rotated MNIST images
def save_rotated_mnist_plot(dataset, num_images=20, filename='rotated_mnist_plot.png',grid_size=(10, 10)):
    # plt.figure(figsize=(10, 2))
    num_images = min(num_images, grid_size[0] * grid_size[1])
    plt.figure(figsize=(grid_size[1] * 1.5, grid_size[0] * 1.5))  # Adjust figure size based on grid size
    
    for i in range(num_images):
        image, label = dataset[i]  # Get image and label
        plt.subplot(1, num_images, i + 1)
        plt.imshow(image.squeeze(), cmap='gray')  # Remove the channel dimension
        plt.title(f"Label: {label}")
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename)  # Save the plot to a file
    print(f"Plot saved to {filename}")


def train():
    # Set seed for reproducibility
    set_seed(42)
    
    # Set device
    device = torch.device('cpu')  # Force CPU
    
    # Move global model to device
    model = selected_model.to(device)
    
    # Load MNIST dataset with augmentation
    train_transform = transforms.Compose([
        # transforms.RandomAffine(0, translate=(0.1, 0.1)),  # Random shift up to 10%
        # transforms.RandomErasing(p=0.2),  # Randomly erase parts of image
        # transforms.CenterCrop(0.4),
        transforms.RandomRotation(degrees=(5, 7)),  # Random rotation up to 5 to 7 degrees
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
    
    # Save augmented samples
    print("### Saving augmented samples...")
    # Save plot of 5 randomly rotated MNIST images
    save_rotated_mnist_plot(train_dataset, filename=os.path.join(os.getcwd(),'Session-5','rotated_mnist_examples.png'))
    print("### Augmented samples saved")
    
    
    # Display model summary
    print("\n### Model Summary:")
    summary(model, (1, 28, 28))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # Train for 1 epoch
    model.train()
    correct = 0
    total = 0
    
    # Modify tqdm to be less verbose
    pbar = tqdm(train_loader, 
                desc='Training',
                ncols=100,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
                dynamic_ncols=False,  # Disable dynamic width updates
                file=sys.stdout        # Ensure output goes to stdout for Actions log
            )
    
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
        # if total % (50 * target.size(0)) == 0:
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}', 
            'accuracy': f'{acc:.2f}%'
        }, refresh=True)
        
    print(f'\n### Final Training Accuracy: {acc:.2f}%')
    
    # Save model with timestamp and accuracy
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = f'model_mnist_{timestamp}_acc{acc:.1f}.pth'
    torch.save(model.state_dict(), save_path)
    print(f'Model saved as {save_path}')
    
if __name__ == '__main__':
    train() 