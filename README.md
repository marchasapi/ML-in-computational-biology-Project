# ML-in-computational-biology-Project
This project focuses on classifying human middle temporal gyrus (MTG) single-nucleus RNA sequencing (snRNA-seq) data using various machine learning and deep learning methods.The dataset includes gene expression profiles for 15,928 samples from eight donors, categorized into 75 distinct cell types. Our approach involves several steps: preprocessing the RNA sequencing data, reducing its dimensions using Principal Component Analysis (PCA), and applying different machine learning algorithms for classification.
We used models like Logistic Regression, Random Forest, k-Nearest Neighbors, Linear Discriminant Analysis, and Support Vector Machines, validated through nested cross-validation for reliable performance measurement. Metrics such as Balanced Accuracy, F1 Score, Sensitivity, and Precision were used to evaluate the models. Additionally, we built a Multilayer Perceptron (MLP) model with TensorFlow and Keras, trying various neural network configurations to find the best performance.By comparing the predictions of different models, we ensure a thorough evaluation of their performance. This project combines traditional machine learning with deep learning and provides an efficient framework for classifying snRNA-seq data.

Project Instructions

Step-by-Step Guide

Step 1 - Preprocessing-R.R

Required files:
human_MTG_2018-06-14_exon-matrix.csv
human_MTG_2018-06-14_genes-rows.csv
human_MTG_2018-06-14_intron-matrix.csv
human_MTG_2018-06-14_samples-columns.csv
These files are large and cannot be uploaded to GitHub. They can be downloaded from the following Google Drive link: https://drive.google.com/drive/folders/1Ae3_XnBHqdjQDkYBvBUQWlCimx5BpWee?usp=sharing
Step 2 - Dimension Reduction

Required file: everything40.csv
This file will be generated after running Step 1. Alternatively, it can also be found at the above Google Drive link.

Step 3 - Machine Learning

Required file: everything40_cleaned_normalized_reduced.csv
This file is provided in the data_preprocessed directory on GitHub. No additional steps are needed; simply run the notebook.

Step 4 - Multilayered Perceptron

Required file: everything40_cleaned_normalized_reduced.csv
This file is the same as the one used in Step 3.

Step 5 - Test Prediction

Required file: everything40_cleaned_normalized_reduced.csv
This file is the same as the one used in Steps 3 and 4.

Additional Resources
Model File: random_forest_model.pkl
This model file is located in the /models directory on the Google Drive. It could not be uploaded to GitHub due to size limitations.
Download it from the following link: https://drive.google.com/drive/folders/1Ae3_XnBHqdjQDkYBvBUQWlCimx5BpWee?usp=sharing
