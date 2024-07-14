## 1: MLP for FashionMNIST

### Overview

This project trains a Multi-Layer Perceptron (MLP) on the FashionMNIST dataset using PyTorch.

### Key Highlights

- **Model Architecture:** Utilizes fully connected layers with ReLU activations.
- **Optimization:** Stochastic Gradient Descent (SGD) is employed to optimize the model parameters.
- **Performance:** Achieves a validation accuracy of at least 82% after 8 epochs.
- **Training and Validation:** Training and validation loss, along with accuracy, are plotted to visualize model performance and convergence.
- **Implementation Details:** Includes data loading, model definition, training loop implementation, and visualization using matplotlib.

## 2: CIFAR-10 CNN with AlexNet Architecture

### Overview

This project implements a Convolutional Neural Network (CNN) using the AlexNet architecture to classify CIFAR-10 images.

### Key Highlights(Workflow)

- **Data Preprocessing:** Applies normalization and augmentation using computed mean and standard deviation values.
- **Dataset Handling:** Loads CIFAR-10 dataset, applies transformations, and creates efficient data loaders for batch processing.
- **Model Architecture:** Implements AlexNet with convolutional layers, ReLU activations, and fully connected layers from scratch, without pre-trained weights.
- **Training Process:** Optimizes model parameters using SGD, tracks training progress with loss per epoch.
- **Validation and Accuracy:** Evaluates model performance on a validation set to prevent overfitting, achieving competitive accuracy on the CIFAR-10 test set.
