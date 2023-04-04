# NaiveBayesClassifier
A Naive Bayes Classifier to get a probabilistic diagnosis of a subset of data containing hippocampal volume. This repository includes code that models a 2D Scatterplot, calculation of mean and variance data, and calculation of the Naive Bayes Classifier accuracy. The test and train data set of hippocampal volume are from: http://www.oasis-brains.org. 

OASIS-hippocampus.csv is the data set containing both the test and train set of hippocampus volume. The data consists of the hippocampal volume, derived from MRI, for elderly subjects, including healthy control subjects and those with mild to moderate dementia.

scatterplot_NBC.py is the hippocampal volume training data set as a 2D scatterplot (right and left hippocampal volume as the two axes). Healthy brains and Dementia positive brains are represented as differennt colors. Heathy brains are green and Dementia positive brains are red on the scatterplot. 

Densityplot_NBC.py is a density plot of both the right and left hippocampal volumes. Healthy brains are colored green in the density plot and Dementia positive brains are colored red. I created density plots for the data to show the distribution of the data for each class. Naive Bayes classification assumes the input features are normally distributed, so density plots can help check if this assumption holds true. If the features are not normally distributed, the classifier may not work as expected in my code "classifier_NBC.py".

classifier_NBC.py calculates the mean and variance of the right and left hippocampus volumes for both healthy and dementia patients, and stores these parameters in a dictionary. It then defines functions to calculate the probability density function and to classify a sample based on the Naive Bayes classifier. The code uses the test data to evaluate the classifier's accuracy, and outputs the final accuracy.
