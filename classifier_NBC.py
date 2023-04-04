import pandas as pd
import numpy as np
import math

df = pd.read_csv("/OASIS-hippocampus.csv")

trainSet = df[df['TrainData'] == 1]

#skip creating diagnosis variable, not needed
healthyClass = trainSet[trainSet['Dementia'] == 0]
dementiaClass = trainSet[trainSet['Dementia'] == 1]

#Calculate the mean and variance of the right and left volume for healthy. Using numpy functions
mean_rightHealthy = healthyClass['RightHippoVol'].mean()
mean_leftHealthy = healthyClass['LeftHippoVol'].mean()
var_rightHealthy = healthyClass['RightHippoVol'].var()
var_leftHealthy = healthyClass['LeftHippoVol'].var()
mean_rightDementia = dementiaClass['RightHippoVol'].mean()
mean_leftDementia = dementiaClass['LeftHippoVol'].mean()
var_rightDementia = dementiaClass['RightHippoVol'].var()
var_leftDementia = dementiaClass['LeftHippoVol'].var()

#Store mean and variance parameters in params dictionary for classification function
params = {'mean_right_healthy': mean_rightHealthy,
          'mean_left_healthy': mean_leftHealthy,
          'var_right_healthy': var_rightHealthy,
          'var_left_healthy': var_leftHealthy,
          'mean_right_dementia': mean_rightDementia,
          'mean_left_dementia': mean_leftDementia,
          'var_right_dementia': var_rightDementia,
          'var_left_dementia': var_leftDementia}

#Function to calculate the probability density function, adapted from lecture slides
def probabilityDensity(x, mean, var):
    return (1 / np.sqrt(2 * math.pi * var)) * np.exp(-(np.power(x - mean, 2) / (2 * var)))

#function to classify a sample based on the Naive Bayes classifier
def classification(rightVol, leftVol, params):
    healthyProb = probabilityDensity(rightVol, params['mean_right_healthy'], params['var_right_healthy']) * probabilityDensity(leftVol, params['mean_left_healthy'], params['var_left_healthy'])
    dementiaProb = probabilityDensity(rightVol, params['mean_right_dementia'], params['var_right_dementia']) * probabilityDensity(leftVol, params['mean_left_dementia'], params['var_left_dementia'])
    healthyProb_given= healthyProb / (healthyProb + dementiaProb)
    dementiaProb_given = dementiaProb / (healthyProb + dementiaProb)

    if healthyProb_given > dementiaProb_given:
        return 0
    else:
        return 1

#test data for comparison
testSet = pd.read_csv('/OASIS-hippocampus.csv')
testSet = testSet[testSet['TrainData'] == 0] #trainData == 0 is the test data 

correctPredictions = 0
totalPredictions = len(testSet)
for index, row in testSet.iterrows():
    rightVol = row['RightHippoVol']
    leftVol = row['LeftHippoVol']
    actual_class = row['Dementia'] #use to compare

    predicted_class = classification(rightVol, leftVol, params)
    if predicted_class == actual_class:
        correctPredictions += 1

#(#correct predictions/#total predictions)
accuracy = correctPredictions / totalPredictions
print("Final Accuracy of Naive Bayes Classifier:", accuracy)
