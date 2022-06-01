# Image to Recipe Project:
The goal of this project is to be able to produce a recipe given an image of food. This project explores a few different deep learning approaches for accomplishing this. 

Dataset: Recipe1m+ dataset

1. Vector similarity approach

This approach works by taking running the food image through a pretrained convolutional neural network. THe image feature vector is taken from the CNN and a cosine similarity is done to find a similar food image, upon which a recipe is retrieved. (knn.py)
   
2. Food image ingredient detection

The idea behind this approach is that a classifier can be trained to learn a list of ingredients from a food recipe, of which then a recipe can be matched. To do this, a classifier is trained by taking the image feature vector, as in above, and using it to detect ingredients by extracting ingredients from the associated recipe and using them as a multi-class one-hot vector label. Unfortunately, this model was not successful. (Garbage in garbage out I suppose)
