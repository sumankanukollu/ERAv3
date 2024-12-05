[![ML Pipeline](https://github.com/sumankanukollu/ERAv3/actions/workflows/ml-pipeline.yml/badge.svg)](https://github.com/sumankanukollu/ERAv3/actions/workflows/ml-pipeline.yml)

# MNIST Classification with CI/CD Pipeline

## Goal:
Make a MNIST based model that has following characteristics:
- Has less than 25000 parameters
- Gets to training accuracy of 95% or more in 1 Epoch
- with a complete CI/CD pipeline using GitHub Actions

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
        ├── rotated_mnist_examples.png
    ├── README.md
    ├── .gitignore # Git ignore rules

```


## Model Architecture

- In `model.py` file we can select the model using a variable, for example `selected_model = MNISTModel_2()`


    ### 1. MNISTModel_1: 
    A basic CNN model with Linear layers 
    - `52k parameters >> Train Accuracy in 1-epoch=94.66% lr=0.01`
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
    ### 2. MNISTModel_2: 
    An optimized CNN model without Linear layer: 
    - In this model I removed the Linear FC layer, as number of parameters in last layer is increased & focused on the receptive field values.
        - Multiple convolutional layers
        - Batch normalization
        - No bias in most layers
    -  `12K Parameters >> Train Accuracy in 1-epoch=95.5% LR=0.1`
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
- Random rotation (5 to 7 degrees)
- Normalization
<!-- - Random affine translations (10%) -->
<!-- - Random erasing (20% probability) -->

### Unit Testcase:
Run 5-unit tests : `pytest -v test_model.py --capture=no`
    
- **Test-case-1:** To test model architecture with respect to parameter count
    - Input shape compatibility (28x28)
    - Model parameter count should be < 25,000
    - Verify model is able to predict only the 10-classes (output shape verification)

- **Test-case-2:** To test model Accuracy
    - Model accuracy should be >95% in 1-epoch
    - Identify `miss-classified` images and save them in `misclassified_grid.png`

- **Test-case-3:** To test model Gradient-flow
    - Load `test_loader` in a batch of `4` and pass the test-input to the model
    - calculate loss and its gradients, and its gradients (mean,std,min and max)
        - Verify that Gradients for each layer are,
            - not contains `NaN values`
            - Gradients are not vanished (mean not all zeros)

- **Test-case-4:** To test model inference with different batch sizes [1, 16, 32, 64, 128]

- **Test-case-5:** Test the robustness of a deep learning model under different input perturbations
    - Load the trained model and set to eval mode.
    - Apply transformations to test data
        - Make an inference on the `original image` to generate prediction
        - Apply a list of three `perturbations` is applied to the input image :
            - Gaussian noise: Add random noise scaled by 0.1.
            - Brightness adjustment: Increase brightness by multiplying the image by 1.2.
            - Contrast adjustment: Decrease contrast by multiplying the image by 0.8.
            -  predicted class for the perturbed image is compared to the original prediction.
    - This scenario is to tests the model's ability to maintain consistent predictions when subjected to slight input modifications, which is crucial for evaluating the model's robustness and generalization capability




    

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
- 20 augmented sample images
- Test results

## Requirements

- Python 3.8+
- PyTorch (CPU version)
- torchvision
- pytest
- tqdm
- torchsummary

## Model Performance

- Parameters: <15,000
- Training time: ~5 minutes on CPU
- Accuracy: >95% after one epoch
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

