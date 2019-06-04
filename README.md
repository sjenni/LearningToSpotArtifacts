# Self-Supervised Feature Learning by Learning to Spot Artifacts [[Project Page]](https://sjenni.github.io/LearningToSpotArtifacts/) 

This repository contains demo code of our CVPR2018 [paper](https://arxiv.org/abs/1806.05024). It contains code for unsupervised training on the unlabeled training set of STL-10 and code for supervised finetuning and evaluation on the labeled datasets. 

## Requirements
The code is based on Python 2.7 and tensorflow 1.12. See requirements.txt for all required packages.

## How to use it

### 1. Setup

- Set the paths to the data and log directories in **globals.py**.
- Run **init_datasets.py** to download and convert the STL-10 dataset.

### 2. Unsupervised Training

- To pre-train the autoencoder run **train_autoencoder_stl10.py**
- To train the classifier and the repair network run **train_stl10.py**

### 3. Transfer & Evaluation

- To finetune the learnt representations run  **fine_tune_stl10.py**
- To evaluate the finetuned classifier run **test_classifier_stl10.py** 

