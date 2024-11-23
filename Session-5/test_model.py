import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from model import selected_model
import glob
import pytest
from datetime import datetime
import numpy as np
import random

def set_seed(seed=42):
    """Set all seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def get_latest_model():
    model_files = glob.glob('model_mnist_*.pth')
    if not model_files:
        raise FileNotFoundError("No model file found")
    latest_model = max(model_files)
    print(f"Using latest model : {latest_model}")
    return latest_model

def load_model_weights(model_path):
    """Safely load model weights with weights_only=True"""
    selected_model.load_state_dict(torch.load(model_path, weights_only=True))
    return selected_model

def test_model_architecture():
    set_seed(42)
    
    # Test 1: Check model parameters count
    total_params = sum(p.numel() for p in selected_model.parameters())
    assert total_params < 25000, f"Model has {total_params} parameters, should be < 25000"
    
    # Test 2: Check input shape compatibility
    test_input = torch.randn(1, 1, 28, 28)
    try:
        output = selected_model(test_input)
        assert output.shape == (1, 10), f"Output shape is {output.shape}, should be (1, 10)"
    except Exception as e:
        pytest.fail(f"Model failed to process 28x28 input: {str(e)}")

def test_model_accuracy():
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = selected_model.to(device)
    
    # Load the latest trained model
    model_path = get_latest_model()
    model = load_model_weights(model_path)
    model.eval()
    
    # Load test dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    
    accuracy = 100. * correct / total
    # import pdb;pdb.set_trace()
    print(f"### Accuracy of the model is : {accuracy}")
    assert accuracy > 90, f"Model accuracy is {accuracy:.2f}%, should be > 90%"

def test_model_gradient_flow():
    set_seed(42)
    model = selected_model
    model.train()
    
    # Create a sample batch
    test_input = torch.randn(4, 1, 28, 28)
    test_target = torch.tensor([0, 1, 2, 3])
    
    # Forward pass
    output = model(test_input)
    loss = F.nll_loss(output, test_target)
    loss.backward()
    
    # Check if gradients exist and are not zero or nan
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Gradient for {name} is None"
            assert not torch.isnan(param.grad).any(), f"Gradient for {name} contains NaN values"
            assert not torch.all(param.grad == 0), f"Gradient for {name} is all zeros"

def test_model_batch_inference():
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = selected_model.to(device)
    model_path = get_latest_model()
    model = load_model_weights(model_path)
    model.eval()
    
    batch_sizes = [1, 16, 32, 64, 128]
    for batch_size in batch_sizes:
        test_input = torch.randn(batch_size, 1, 28, 28).to(device)
        with torch.no_grad():
            output = model(test_input)
            
        assert output.shape == (batch_size, 10), \
            f"Failed to process batch size {batch_size}. Output shape: {output.shape}"
        
        # Check if probabilities sum to 1 for each prediction
        probs = torch.exp(output)
        prob_sums = probs.sum(dim=1)
        assert torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-6), \
            f"Output probabilities don't sum to 1 for batch size {batch_size}"

def test_model_robustness():
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = selected_model.to(device)
    model_path = get_latest_model()
    model = load_model_weights(model_path)
    model.eval()
    
    # Load a single test image
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1)
    original_image, label = next(iter(test_loader))
    original_image = original_image.to(device)
    
    with torch.no_grad():
        # Get original prediction
        original_output = model(original_image)
        original_pred = original_output.argmax(dim=1)
        
        # Test with different perturbations
        perturbations = [
            ('gaussian_noise', original_image + 0.1 * torch.randn_like(original_image)),
            ('brightness', original_image * 1.2),
            ('contrast', original_image * 0.8)
        ]
        
        consistent_predictions = 0
        for perturb_name, perturbed_image in perturbations:
            perturbed_output = model(perturbed_image)
            perturbed_pred = perturbed_output.argmax(dim=1)
            
            if perturbed_pred == original_pred:
                consistent_predictions += 1
                
        # Model should maintain prediction for at least 2 out of 3 perturbations
        assert consistent_predictions >= 2, \
            f"Model predictions are not consistent under perturbations. " \
            f"Consistent predictions: {consistent_predictions}/3"

if __name__ == '__main__':
    pytest.main([__file__]) 