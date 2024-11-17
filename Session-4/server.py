from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import json
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import MNISTFeedForward, MNISTCNN, get_optimizer, create_model
import threading
import time
from datetime import datetime
from torchsummary import summary
from io import StringIO
import sys

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Store data for both models
models_data = {
    1: {
        'status': 'idle',
        'losses': [],
        'accuracies': [],
        'iterations': [],
        'metrics': {
            'training_time': '-',
            'final_train_acc': '-',
            'final_val_acc': '-'
        }
    },
    2: {
        'status': 'idle',
        'losses': [],
        'accuracies': [],
        'iterations': [],
        'metrics': {
            'training_time': '-',
            'final_train_acc': '-',
            'final_val_acc': '-'
        }
    }
}

def get_model_summary(model):
    """Capture model summary as a string"""
    # Create StringIO object to capture the summary
    summary_str = StringIO()
    original_stdout = sys.stdout
    sys.stdout = summary_str
    
    # Print model summary
    summary(model, (1, 28, 28))  # MNIST input size
    
    # Restore stdout and get the summary string
    sys.stdout = original_stdout
    return summary_str.getvalue()

def train_model(model_num, config):
    """
    Train the model in a separate thread
    """
    try:
        # Reset model data
        models_data[model_num] = {
            'status': 'training',
            'losses': [],
            'accuracies': [],
            'iterations': [],
            'metrics': {
                'training_time': '-',
                'final_train_acc': '-',
                'final_val_acc': '-'
            }
        }

        start_time = time.time()

        # Device configuration
        device = torch.device('cpu')

        # Create model based on type
        if config['model_type'] == 'fnn':
            model = create_model(
                'fnn',
                channels=config['channels'],
                dropout_rate=config['dropout']
            ).to(device)
        else:  # CNN model
            model = create_model(
                'cnn',
                channels=config['channels'],
                dropout_rate=config['dropout']
            ).to(device)

        # Get model summary
        model_summary = get_model_summary(model)
        models_data[model_num]['model_summary'] = model_summary
        
        # Print model summary to console
        logger.info(f"\nModel {model_num} Summary:")
        logger.info("=" * 50)
        logger.info(model_summary)
        logger.info("=" * 50)

        # Print model architecture
        logger.info(f"\nModel Architecture:")
        logger.info(str(model))
        
        # Print parameter count
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"\nTotal Parameters: {total_params:,}")
        logger.info(f"Trainable Parameters: {trainable_params:,}")
        
        # Print configuration
        logger.info("\nTraining Configuration:")
        logger.info(f"Model Type: {config['model_type']}")
        logger.info(f"Optimizer: {config['optimizer']}")
        logger.info(f"Learning Rate: {config['learning_rate']}")
        logger.info(f"Batch Size: {config['batch_size']}")
        logger.info(f"Epochs: {config['epochs']}")
        logger.info(f"Dropout Rate: {config['dropout']}")
        if config['model_type'] == 'fnn':
            logger.info(f"Hidden Neurons: {config['channels']}")
        else:
            logger.info(f"Kernel Sizes: {config['channels']}")
        logger.info("=" * 50 + "\n")

        # MNIST dataset
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
        test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)

        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = get_optimizer(
            config['optimizer'],
            model.parameters(),
            config['learning_rate']
        )

        # Training loop
        final_train_acc = 0
        final_val_acc = 0
        
        logger.info(f"Starting training for Model {model_num}")
        
        for epoch in range(config['epochs']):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for i, (images, labels) in enumerate(train_loader):
                images = images.to(device)
                labels = labels.to(device)
                
                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                running_loss += loss.item()
            
            # Calculate epoch metrics
            epoch_loss = running_loss / len(train_loader)
            epoch_acc = 100 * correct / total
            
            # Store metrics for each epoch
            models_data[model_num]['losses'].append(epoch_loss)
            models_data[model_num]['accuracies'].append(epoch_acc)
            models_data[model_num]['iterations'].append(epoch + 1)
            
            final_train_acc = epoch_acc

            # Validation
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for images, labels in test_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                
                final_val_acc = 100 * correct / total
                
            logger.info(f"Epoch [{epoch+1}/{config['epochs']}] - "
                       f"Train Loss: {epoch_loss:.4f}, "
                       f"Train Acc: {epoch_acc:.2f}%, "
                       f"Val Acc: {final_val_acc:.2f}%")

        end_time = time.time()
        training_time = f"{end_time - start_time:.2f}s"

        # Update metrics
        models_data[model_num]['status'] = 'completed'
        models_data[model_num]['metrics']['training_time'] = training_time
        models_data[model_num]['metrics']['final_train_acc'] = f"{final_train_acc:.2f}%"
        models_data[model_num]['metrics']['final_val_acc'] = f"{final_val_acc:.2f}%"

    except Exception as e:
        logger.error(f"Error training model {model_num}: {str(e)}")
        models_data[model_num]['status'] = 'error'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/train', methods=['POST'])
def train():
    try:
        data = request.json
        model_num = data['model_num']
        config = data['config']
        
        # Start training in a separate thread
        thread = threading.Thread(
            target=train_model,
            args=(model_num, config)
        )
        thread.start()
        
        return jsonify({'status': 'success'})
    except Exception as e:
        logger.error(f"Error starting training: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/get_data')
def get_data():
    return jsonify({
        'model1': models_data[1],
        'model2': models_data[2]
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001) 