# Code of paper "Pseudo-likelihood produces associative memories able to generalize, even for asymmetric couplings"
Francesco D'Amico, Dario Bocchi, Luca Maria Del Bono, Saverio Rossi, Matteo Negri

ArXiv: https://arxiv.org/abs/2507.05147

The code released has been produced to run simulations for pseudolikelihood training of a binary, two-bodies interaction model.
This code has three built-in custom datasets: random binary data and random features binary data, and binarized MNIST data.

---

## Attention: new version available
If you are interested in pseudolikelihood training and random/random-features datasets, we have made a cleaner, optimized, more general and library-like repository:

https://github.com/Francill99/PL

This repository remains for reference for the code and data specific of the paper 

## Installation

Clone the repository:

```bash 
git clone https://github.com/Francill99/PseudoLikelihood_Analysis.git
```
Install:

```bash 
pip install -e .
```
Dependencies: to create the same conda (or miniconda) environment we used

```bash 
conda env create -f environment.yml
conda activate your_env_name
```

---

## Structure of the repository

In "Pseudolikelihodd_Analysis" folder there are codes used. In "Graphs" folder there are data and code to reproduce the plots in the paper.

---

## Binarized MNIST

The entire experiment for MNIST data, including the procedure to generate the binarized dataset as in M. Belyaev, A. Velichko, "Classification of handwritten digits using the hopfield network", can be found inside the notebook ""

---
## Basic Example

Run a single Pseuodlikelihood training with random data for 400 steps:

```bash
chmod +x simple_training.sh
./simple_training.sh
```
Parameters can be changed inside file simple_training.sh. It will be created a file training_log.txt containing the log of 
the training process, and at end of training in folder "savings" it will be saved a checkpoint of the model at last step.
Metrics in the log file corresponds to final overlaps of dynamics with respect to training data, features and generalization data.
Set D=0 to simulate model without features. In that case all final overlaps are related to training data.


