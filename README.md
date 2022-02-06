# Neural Networks with PyTorch (Beginners)

Introduction 
===========

In this project we will go through some basic data pre-processing with Python, visualize our dataset with Seaborn and Matplotlib, split it with sklearn, and train a simple Multi-Layer Perceptron (MLP) using PyTorch to solve the [WIDS22 Challenge](https://www.kaggle.com/c/widsdatathon2022) prediction problem. The implementation is based on the Kaggle [notebook](https://www.kaggle.com/azdeck/wids22-neural-networks-with-pytorch-beginners).

Getting Started
============
We recommend using a python virtual environment

```
python3 -m venv WIDS python=3.10
```

Install the requirements

```
pip3 install -r requirements.txt
```

Train the model

```
python train.py
```

You can visualize the dataset by passing the command ``-v True`` and change the epoch number by setting ``-e <number_of_epochs>``.

The training statistics can be found on tensorboard log directory and can be accessed by running:

```
tensorboard --logdir runs
```
