# Neural Network from Scratch

## Overview

This project implements a neural network from scratch using NumPy, providing a hands-on understanding of core concepts in deep learning. The network includes various layers such as convolutional, fully connected, pooling, and activation functions. The project supports forward and backward passes, gradient checking, and includes L2 regularization.

## Features

- **Layers Implemented:**
  - Convolution
  - MaxPooling
  - ReLU
  - Fully Connected

- **Regularization:**
  - L2 Regularization
- **Loss Functions:**
  - Softmax Cross-Entropy with Logits

- **Training and Evaluation:**
  - Functions for training, evaluating, and saving/loading models
- **Gradient Checking:**
  - Numerical gradient checking for verification

## Getting Started

### Prerequisites

- Python 3.x
- NumPy
- Other dependencies listed in 'requirements.txt'


### Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/YourUsername/neural_net_from_scratch.git
   cd neural_net_from_scratch

   ```
2. **Set up a Virtual Environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```
4. Before running the project, you need to build necessary Cython extensions which are used for performance
optimization and efficient memory handling.
   ```bash
   python setup_cython.py build_ext --inplace
   ```
### Usage

Training the Network:
   ```bash
   python train.py
   ```
The 'train.py' script sets up the network, gets the MNIST dataset, 
trains the model and evaluates its performance. 
You can run 'check_grads.py' to verify the correctness of gradients in each type of layer. 



### Notes

- Update 'config' in 'train.py' with your specific training hyperparameters (e.g., batch size, learning rate).
- The 'check_grads.py' file is used for validating gradient computations against numerical gradients.
- Change the architecture of network based on your problem and needs (what works for one problem
most likely will not work for the other one)

### Contribution
Feel free to open issues or submit pull requests. Contributions are welcome to enhance functionality and performance.


