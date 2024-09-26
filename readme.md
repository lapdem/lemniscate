# Project Lemniscate

This project explores the large with limit of artificial neural networks. In particular it aims to establish the behaviour of a neural network during training.

The code is divided into three modules.

## Experiment
This module contains everything relating to actually running a given experiment. 
It reads the experiment configuration file, runs the analytic calcualtions and sets up an ensemble actual neural network to verify the results. Outputs (currently images of the trainig process) are saved into a specified output folder

## Analytic
This module contains all code relating to the analytic calculations of a neural networks behaviour in the wide width limit. An introduction to the topic can be found in the book [The Principles of Deep Learning Theory](https://arxiv.org/abs/2106.10165) and a detailed documentation about the calculations performed here can be found in the theory folder of this project.

## Numeric 
This module contains the code used to set up and run an actual Neural Network to verify the analytic calculations. It is loosly based on the book [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/) and it's [reference implementation](https://github.com/mnielsen/neural-networks-and-deep-learning?tab=readme-ov-file). Currently training is doneby gradient descent using simple forward difference numeric differentiation.


Code that gets reused across the three sections can be found in a separate util module