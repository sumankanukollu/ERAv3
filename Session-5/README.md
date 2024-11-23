[![ML Pipeline](https://github.com/sumankanukollu/ERAv3/actions/workflows/ml-pipeline.yml/badge.svg)](https://github.com/sumankanukollu/ERAv3/actions/workflows/ml-pipeline.yml)

# MNIST Classification with CI/CD Pipeline

This project implements a Convolutional Neural Network (CNN) for MNIST digit classification with a complete CI/CD pipeline using GitHub Actions. The model achieves >95% accuracy in just one epoch while maintaining a small parameter count < 25K.

## Project Structure 
```
ERAV3
├── README.md
├── Session-1
├── Session-2
├── Session-3
├── Session-4
├── Session-5
└── .github/
    └── workflows/
        └── ml-pipeline.yml # CI/CD pipeline configuration


Session-5
    ├── requirements.txt # Project dependencies
    ├── model.py # Neural network architecture definitions
    ├── train.py # Training script with data augmentation
    ├── test_model.py # Test suite for model validation
    ├── augmented_samples
        ├── augmented_samples_grid.png
    ├── README.md
    ├── .gitignore # Git ignore rules

```


## Model Architecture

The project includes two or more model architectures:
1. `MNISTModel`: A basic CNN model with ~52k parameters
    ```
    ### Model Summary:
    ----------------------------------------------------------------
            Layer (type)               Output Shape         Param #
    ================================================================
                Conv2d-1            [-1, 8, 28, 28]              80
                Conv2d-2           [-1, 16, 14, 14]           1,168
                Linear-3                   [-1, 64]          50,240
                Linear-4                   [-1, 10]             650
    ================================================================
    Total params: 52,138
    Trainable params: 52,138
    Non-trainable params: 0
    ----------------------------------------------------------------
    Input size (MB): 0.00
    Forward/backward pass size (MB): 0.07
    Params size (MB): 0.20
    Estimated Total Size (MB): 0.27
    ----------------------------------------------------------------
    ```

2. `MNISTModel_2`: An optimized CNN with:
   - Multiple convolutional layers
   - Batch normalization
   - No bias in most layers
   - Receptive field calculation comments
   ```
    ### Model Summary:
    ----------------------------------------------------------------
            Layer (type)               Output Shape         Param #
    ================================================================
                Conv2d-1            [-1, 8, 26, 26]              72
                ReLU-2            [-1, 8, 26, 26]               0
        BatchNorm2d-3            [-1, 8, 26, 26]              16
                Conv2d-4           [-1, 16, 24, 24]           1,152
                ReLU-5           [-1, 16, 24, 24]               0
        BatchNorm2d-6           [-1, 16, 24, 24]              32
            MaxPool2d-7           [-1, 16, 12, 12]               0
                Conv2d-8           [-1, 16, 10, 10]           2,304
                ReLU-9           [-1, 16, 10, 10]               0
        BatchNorm2d-10           [-1, 16, 10, 10]              32
            Conv2d-11             [-1, 16, 8, 8]           2,304
                ReLU-12             [-1, 16, 8, 8]               0
        BatchNorm2d-13             [-1, 16, 8, 8]              32
            Conv2d-14             [-1, 16, 6, 6]           2,320
                ReLU-15             [-1, 16, 6, 6]               0
        BatchNorm2d-16             [-1, 16, 6, 6]              32
            Conv2d-17             [-1, 16, 4, 4]           2,304
                ReLU-18             [-1, 16, 4, 4]               0
        BatchNorm2d-19             [-1, 16, 4, 4]              32
            Conv2d-20             [-1, 10, 4, 4]             160
                ReLU-21             [-1, 10, 4, 4]               0
        BatchNorm2d-22             [-1, 10, 4, 4]              20
            Conv2d-23             [-1, 10, 1, 1]           1,600
    ================================================================
    Total params: 12,412
    Trainable params: 12,412
    Non-trainable params: 0
    ----------------------------------------------------------------
    Input size (MB): 0.00
    Forward/backward pass size (MB): 0.44
    Params size (MB): 0.05
    Estimated Total Size (MB): 0.49
    ----------------------------------------------------------------
    ```

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
    ```pytest test_model.py -v --capture=no```


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
- 30 augmented sample images
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

