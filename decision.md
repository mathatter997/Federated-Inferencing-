We planned to use Score-Based models for Federated Inferencing because differentials are linear and adding them is a natural operation.
We found `https://github.com/yang-song/score_sde` which provides a python notebook for training and running Score-Based SDE diffusion and modify it to train a suite of different models.
Because of the size of these models (>600MB) all models were all kept on google drive, where they could be quickly mounted into a colab notebook.
We created a `inference.ipynb` to seperately generate the output. Because of the length of runtimes, the Colab session would abort and delete delete the runtime session. 
This was very annoying and inferencing had to be repeated on several occasions. We ended up generating the output in batches of digits at a time to avoid Colab deleting all of our progress.
Once the images and score functions were obtained, the rest of the analysis went smoothly. We divided the work in half: one of us worked on the FID and Inception scores, while the other worked
on generating the score-function stastics. 


