import torch
import torch.nn as nn

class MNISTFeedForward(nn.Module):
    def __init__(self, channels, dropout_rate=0.2):
        super(MNISTFeedForward, self).__init__()
        self.flatten = nn.Flatten()
        
        self.layers = nn.Sequential(
            nn.Linear(28 * 28, channels[0]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(channels[0], channels[1]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(channels[1], channels[2]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(channels[2], 10)
        )
        
    def forward(self, x):
        x = self.flatten(x)
        return self.layers(x)

class MNISTCNN(nn.Module):
    def __init__(self, channels, dropout_rate=0.2):
        """
        channels: list of 4 integers for channel sizes of each conv layer
        kernel_size is fixed at 3
        """
        super(MNISTCNN, self).__init__()
        
        # Fixed kernel size
        kernel_size = 3
        
        # Convolutional layers
        self.conv_layers = nn.Sequential(
            # First conv layer: 1 -> channels[0]
            nn.Conv2d(1, channels[0], kernel_size=kernel_size, padding=kernel_size//2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout_rate),
            
            # Second conv layer: channels[0] -> channels[1]
            nn.Conv2d(channels[0], channels[1], kernel_size=kernel_size, padding=kernel_size//2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout_rate),
            
            # Third conv layer: channels[1] -> channels[2]
            nn.Conv2d(channels[1], channels[2], kernel_size=kernel_size, padding=kernel_size//2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout_rate),
            
            # Fourth conv layer: channels[2] -> channels[3]
            nn.Conv2d(channels[2], channels[3], kernel_size=kernel_size, padding=kernel_size//2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # Global average pooling
            nn.Dropout2d(dropout_rate)
        )
        
        # Fully connected layer
        self.fc = nn.Linear(channels[3], 10)
        
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x

def create_model(model_type, channels=None, dropout_rate=0.2):
    """
    Factory function to create either FNN or CNN model
    """
    if model_type.lower() == 'fnn':
        return MNISTFeedForward(channels, dropout_rate)
    elif model_type.lower() == 'cnn':
        return MNISTCNN(channels, dropout_rate)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def get_optimizer(optimizer_name, model_parameters, learning_rate):
    """
    Returns the specified optimizer with given learning rate
    """
    if optimizer_name.lower() == 'sgd':
        return torch.optim.SGD(model_parameters, lr=learning_rate)
    elif optimizer_name.lower() == 'momentum':
        return torch.optim.SGD(model_parameters, lr=learning_rate, momentum=0.9)
    elif optimizer_name.lower() == 'nag':
        return torch.optim.SGD(model_parameters, lr=learning_rate, momentum=0.9, nesterov=True)
    elif optimizer_name.lower() == 'adam':
        return torch.optim.Adam(model_parameters, lr=learning_rate)
    elif optimizer_name.lower() == 'adagrad':
        return torch.optim.Adagrad(model_parameters, lr=learning_rate)
    elif optimizer_name.lower() == 'rmsprop':
        return torch.optim.RMSprop(model_parameters, lr=learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}") 