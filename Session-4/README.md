# MNIST Neural Network Visualization

This project implements a 4-layer neural network for MNIST digit classification with real-time training visualization.

## Project Structure 

## Features

1. Interactive Web Interface
   - Side-by-side model configuration
   - Real-time training visualization
   - Model performance comparison
   - Dynamic model architecture selection (FFNN/CNN)

2. Model Types
   - Feed Forward Neural Network (FFNN)
     * Configurable hidden layer neurons
     * Default: 512->256->128->10
   - Convolutional Neural Network (CNN)
     * Configurable channel sizes
     * Fixed 3x3 kernels
     * Default: 16->32->64 channels

3. Training Parameters
   - Multiple optimizer options (SGD, Adam, RMSprop, etc.)
   - Configurable learning rate
   - Adjustable batch size
   - Variable epochs
   - Dropout rate control

4. Real-time Visualization
   - Training accuracy curves
   - Loss curves
   - Model summaries
   - Performance metrics comparison

## Setup and Installation

1. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

2. Start the Flask server:
```python server.py```


3. Open your browser and navigate to:
   ```http://127.0.0.1:5001/```


## Usage

1. Configure Model 1:
   - Select model type (FFNN/CNN)
   - Set architecture parameters
   - Configure training parameters

2. Configure Model 2:
   - Repeat the same process with different settings

3. Start Training:
   - Click "Train Model" buttons
   - Watch real-time progress
   - Compare performance metrics

## Technology Stack

- Python 3.8+
- PyTorch
- Flask
- HTML/CSS/JavaScript
- Plotly.js
- Multi-threading

## System Requirements

- Operating System: macOS (10.15+)
- RAM: 8GB minimum
- CPU: Multi-core processor
- Python environment with pip

## Notes

- Training is performed on CPU on MAC M1
- Multi-threaded implementation allows training multiple models simultaneously
- Real-time visualization updates every 500ms
- Model summaries show architecture details and parameter counts

# Demo Video:
[![Demo Video](https://img.youtube.com/vi/fD696FalGZk/0.jpg)](https://youtu.be/fD696FalGZk?list=PLqYDykjFMcnE6_MKqsWsrl_hsUkVxPTdP)
