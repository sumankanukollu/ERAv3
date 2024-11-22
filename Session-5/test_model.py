import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from model import MNISTModel,MNISTModel_2
import glob
import pytest

def get_latest_model():
    model_files = glob.glob('model_mnist_*.pth')
    if not model_files:
        raise FileNotFoundError("No model file found")
    latest_model = max(model_files)
    print(f"Using latest model : {latest_model}")
    return latest_model

def test_model_architecture():
    model = MNISTModel_2()
    
    # Test 1: Check model parameters count
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params < 100000, f"Model has {total_params} parameters, should be < 100000"
    
    # Test 2: Check input shape compatibility
    test_input = torch.randn(1, 1, 28, 28)
    try:
        output = model(test_input)
        assert output.shape == (1, 10), f"Output shape is {output.shape}, should be (1, 10)"
    except Exception as e:
        pytest.fail(f"Model failed to process 28x28 input: {str(e)}")

def test_model_accuracy():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MNISTModel_2().to(device)
    
    # Load the latest trained model
    model_path = get_latest_model()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Load test dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000)
    
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
    print(f"Accuracy of the model is : {accuracy}")
    assert accuracy > 90, f"Model accuracy is {accuracy:.2f}%, should be > 90%"

if __name__ == '__main__':
    pytest.main([__file__]) 