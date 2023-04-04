#sources : https://www.datacamp.com/tutorial/pandas-read-csv

import matplotlib.pyplot as plot
import pandas as pd

#Read data from /OASIS-hippocampus.csv, NOTE: csv file has to be uploaded to jupyter if using cloud version(this resets everytime runtime is disabled so remember to upload)
df = pd.read_csv('/OASIS-hippocampus.csv')

#Diagnosis for dementia dataframe(pandas)
diagnosis = df['Dementia']

#dataframe to determine if healthy/dementia
dementia = df[diagnosis == 1]
healthy = df[diagnosis == 0]

#Instead HippoVol can just pull directly from csv(faster)
plot.scatter(healthy['RightHippoVol'], healthy['LeftHippoVol'], color='green', label='Healthy')
plot.scatter(dementia['RightHippoVol'], dementia['LeftHippoVol'], color='red', label='Dementia')

#grid background
plot.grid(True, linestyle='-')

#labels and title
plot.xlabel('Right Hippocampus Volume')
plot.ylabel('Left Hippocampus Volume')
plot.title('Classificiation of Dementia on Left vs Right Hippocampus Volume')
plot.legend()

plot.show()
