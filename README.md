# ML-in-computational-biology-Project
This project focuses on classifying human middle temporal gyrus (MTG) single-nucleus RNA sequencing (snRNA-seq) data using various machine learning and deep learning methods.The dataset includes gene expression profiles for 15,928 samples from eight donors, categorized into 75 distinct cell types. Our approach involves several steps: preprocessing the RNA sequencing data, reducing its dimensions using Principal Component Analysis (PCA), and applying different machine learning algorithms for classification.
We used models like Logistic Regression, Random Forest, k-Nearest Neighbors, Linear Discriminant Analysis, and Support Vector Machines, validated through nested cross-validation for reliable performance measurement. Metrics such as Balanced Accuracy, F1 Score, Sensitivity, and Precision were used to evaluate the models. Additionally, we built a Multilayer Perceptron (MLP) model with TensorFlow and Keras, trying various neural network configurations to find the best performance.By comparing the predictions of different models, we ensure a thorough evaluation of their performance. This project combines traditional machine learning with deep learning and provides an efficient framework for classifying snRNA-seq data.

How to run the code:

Run the notebooks in the order they are displayed (Step 1, then Step 2, then Step 3, then Step 4, then Step 5)

In order to run the Project - Step 1 - Preprocessing-R.R notebook the following csv files are needed: human_MTG_2018-06-14_exon-matrix.csv , human_MTG_2018-06-14_genes-rows.csv , human_MTG_2018-06-14_intron-matrix.csv , human_MTG_2018-06-14_samples-columns.csv These files due to their size can not be uploadwd on github and can be found at https://drive.google.com/drive/folders/1Ae3_XnBHqdjQDkYBvBUQWlCimx5BpWee?usp=sharing

In order to run the Project - Step 2 - Dimension reduction.ipynb  the everything40.csv is needed. This file will occur if you run Project - Step 1 - Preprocessing-R.R . Else it can be also be found at the previous drive link.

In order to run Project - Step 3 - Machine learning.ipynb , Project - Step 4 - Multilayered Perceptron.ipynb and Project - Step 5 - Test prediction.ipynb the everything40_cleaned_normalized_reduced.csv is needed. This csv is provided in this github in the data_preprocessed file. You do not need to do anything, just run the notebooks.

In the following drive the random_forest_model.pkl can be found in the /models directory (It could not be uploaded on github also due to size limitations):
https://drive.google.com/drive/folders/1Ae3_XnBHqdjQDkYBvBUQWlCimx5BpWee?usp=sharing
