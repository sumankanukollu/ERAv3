# MNIST Classification with CI/CD Pipeline

This project implements a Convolutional Neural Network (CNN) for MNIST digit classification with a complete CI/CD pipeline using GitHub Actions. The model achieves >95% accuracy in just one epoch while maintaining a small parameter count < 25K.

## Project Structure 
```
├── model.py # Neural network architecture definitions
├── train.py # Training script with data augmentation
├── test_model.py # Test suite for model validation
├── requirements.txt # Project dependencies
├── .gitignore # Git ignore rules
└── .github/
└── workflows/
└── ml-pipeline.yml # CI/CD pipeline configuration
```


## Model Architecture

The project includes two model architectures:
1. `MNISTModel`: A basic CNN with ~25k parameters
2. `MNISTModel_2`: An optimized CNN with:
   - Multiple convolutional layers
   - Batch normalization
   - No bias in most layers
   - Receptive field calculation comments

## Features

### Data Augmentation
The training pipeline includes several augmentation techniques:
- Random rotation (±15 degrees)
- Random affine translations (10%)
- Random erasing (20% probability)
- Normalization

### Automated Testing
The test suite (`test_model.py`) verifies:
- Model parameter count (< 25,000)
- Input shape compatibility (28x28)
- Output shape verification (10 classes)
- Model accuracy (>90% requirement)

### CI/CD Pipeline
The GitHub Actions workflow:
- Runs on Ubuntu latest
- Uses CPU-only PyTorch
- Caches pip dependencies
- Trains the model
- Runs all tests
- Archives model weights and augmented samples

## Getting Started

### Local Development

1. Create a virtual environment:
    ```
    bash
    python -m venv venv
    source venv/bin/activate # On Windows: venv\Scripts\activate
    ```


2. Install dependencies:
    ```
    pip install -r requirements.txt
    ```

3. Train the model:
    ```python train.py```

4. Run Tests:
    ```pytest test_model.py -v```


### GitHub Integration

1. Fork/clone this repository
2. Push your changes
3. GitHub Actions will automatically:
   - Train the model
   - Run tests
   - Save artifacts

## Artifacts

The pipeline generates and stores:
- Trained model weights (`model_mnist_[timestamp]_acc[accuracy].pth`)
- 100 augmented sample images
- Test results

## Requirements

- Python 3.8+
- PyTorch (CPU version)
- torchvision
- pytest
- tqdm
- torchsummary

## Model Performance

- Parameters: <25,000
- Training time: ~5 minutes on CPU
- Accuracy: >90% after one epoch
- Input size: 28x28 grayscale images
- Output: 10 classes (digits 0-9)

## Directory Structure Details

### augmented_samples/
- Contains 100 augmented training samples
- First 5 samples are tracked in git
- Others are ignored but included in CI/CD artifacts

### .github/workflows/
- Contains CI/CD pipeline configuration
- Runs on every push
- Artifacts retained for 90 days

## Notes

- The model is designed for CPU training
- All tests must pass for successful CI/CD
- Augmented samples help verify data pipeline
- Model weights include timestamp and accuracy in filename

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

