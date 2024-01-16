Due to the size of the diffusion models, most the the project was done on Colab using a shared drive.

`Diffusers_SDE.ipynb` modified from the original in `https://github.com/yang-song/score_sde`. Using the MNIST dataset we trained a centralized control model. We split MNIST into 10 homogeneous dataset and trained 10 agents. We split the MNIST dataset into 10 heterogeneous datasets and trained 10 agents.

We generated outputs using `inference.ipynb`. We inferenced 10 samples for each digit across all three regimes. 


