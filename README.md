# MNIST Digit Classification System

## Overview
This project implements a digit classification system using the MNIST dataset. The model is trained to recognize handwritten digits (0-9) using machine learning techniques, specifically deep learning with neural networks.

## Features
- Preprocessing of MNIST dataset
- Training of a neural network for digit classification
- Evaluation of model performance
- Deployment of the model for real-time digit prediction

## Technologies Used
- Python
- TensorFlow/Keras
- NumPy
- Matplotlib
- Jupyter Notebook

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/vigkrishna/MNIST-DIGIT-CLASSIFICATION-SYSTEM.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the training script:
   ```bash
   python train.py
   ```

## Usage
- Train the model by running `train.py`.
- Evaluate the model using `evaluate.py`.
- Use `predict.py` to classify new handwritten digit images.

## Dataset
The MNIST dataset consists of 60,000 training images and 10,000 test images of handwritten digits. It is publicly available at [Yann LeCun's website](http://yann.lecun.com/exdb/mnist/).

## Model Architecture
- Input Layer: 28x28 flattened to 784 nodes
- Hidden Layers: Fully connected layers with ReLU activation
- Output Layer: 10 nodes with softmax activation
- Loss Function: Categorical Crossentropy
- Optimizer: Adam

## Results
The trained model achieves an accuracy of approximately **98%** on the test dataset.

## Future Enhancements
- Implement Convolutional Neural Networks (CNNs) for improved accuracy
- Develop a web-based UI for user interaction
- Deploy the model as an API

## License
This project is licensed under the MIT License.

## Acknowledgments
- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
- TensorFlow and Keras community

