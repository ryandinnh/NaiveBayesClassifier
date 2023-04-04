import matplotlib.pyplot as plot
import numpy as np #approved via Piazza
import pandas as pd

#Read data from /OASIS-hippocampus.csv
df = pd.read_csv('/OASIS-hippocampus.csv')

#Diagnosis for dementia dataframe(pandas)
diagnosis = df['Dementia']

#dataframe to determine if healthy/dementia
dementia = df[diagnosis == 1]
healthy = df[diagnosis == 0]

#Creating environment for plots
plot.figure(figsize=(12,12)) #dimensions dont matter going to use tight adjust below

#Density plot for left hippocampus
#creating subplot index 1 for lefthippvol
plot.subplot(2, 2, 1) #args(rows, col, index)

#Creating histogram vals. 
#additional parameters: alpha for opaqueness 
plot.hist(healthy["LeftHippoVol"], color="green", alpha=0.75, label="Healthy")
plot.hist(dementia["LeftHippoVol"], color="red", alpha= 0.75, label="Dementia")

#Labels and Titles of graph
plot.xlabel("Left Hippocampus Volume")
plot.ylabel("Density")
plot.title("Density Plot of Left Hippocampus Volume on Dementia Classification")
plot.legend()
plot.grid(True, linestyle='-')

#Density plot for right hippocampus
#creating a subplot index 2 for righthippoval
plot.subplot(2, 2, 2) #args(rows, col, index)

plot.hist(healthy["RightHippoVol"], color ="green", alpha=0.75, label ="Healthy")
plot.hist(dementia["RightHippoVol"], color="red", alpha=0.75, label="Dementia")

plot.xlabel("Right Hippocampus Volume")
plot.ylabel("Density")
plot.title("Density Plot of Right Hippocampus Volume on Dementia Classification")
plot.legend()
plot.grid(True, linestyle='-')
plot.tight_layout()

plot.show()
