# FederatedInference: Federated Inferencing for Score-Based Diffusion Models

By: Luke Braithwaite (lb2027@cam.ac.uk) and Matthew Hattrup (mh2236@cam.ac.uk)

Federated Learning has emerged as a means to train models over decentralized data. 
In the real world, data often cannot be centralized due to costs, privacy concerns, or regulation making centralized training in these cases impossible. Federated Learning is a potential solution for training in these contexts, whereby a central model learns a global distribution by aggregating training over isolated datasets. 
In this paper, we propose FederatedInference framework, whereby models are pretrained on seperate datasets and their predictions are combined for collective inferencing. 
We adapt Score-Based diffusion models to a generative framework and experiment with different regimes where the agents are trained on homogeneous and heterogenous datasets. 
We show that in both the homogeneous and heterogeneous regimes, federated inferencing performs similarily. 
We also show that our FederatedInference framework is more inconsistent than a centralized approach at producing correct digits, but when it does, the images are of a higher quality than a centralized approach. 
We suggest possible remedies in the discussion. 
In the future, FederatedInference could be a useful tool for collective inferencing across models trained on isolated datasets.

## Overview
Due to the size of the diffusion models, most the the project was done on Colab using a shared drive.

`Diffusers_SDE.ipynb` modified from the original in `https://github.com/yang-song/score_sde`. Using the MNIST dataset we trained a centralized control model. We split MNIST into 10 homogeneous dataset and trained 10 agents. We split the MNIST dataset into 10 heterogeneous datasets and trained 10 agents.

We generated outputs using `inference.ipynb`. We inferenced 10 samples for each digit across all three regimes. 

## Running this code
We ran this code on Google Collab and used a T4 GPU for the training and inference.
By default, each notebook saves the outputs to google drive, that can be adjusted as you see fit by modifying the code.


## Code structure
In this repository, you shall find 3 jupyter notebooks.

 1. `Diffusers_SDE.ipynb`⁠: this notebook is used to pretrain a score based diffusion model using huggingface's diffusers library. Adapted from https://github.com/yang-song/score_sde which is the original score-based diffusion code.
 2. ⁠⁠`inference.ipynb`: given a series of pretrained diffusion models perform federated inference on them. We used [torchdiffeq](https://github.com/rtqichen/torchdiffeq) as our SDE solver due to it interoperability with the PyTorch ecosystem.
 3. `⁠Analysis.ipynb`⁠: performs the analysis on samples generated using federated inferencing. For the MNIST feature extractor we used the pretrained model weights from https://github.com/sundyCoder/IS_MS_SS that was based on ResNet.




