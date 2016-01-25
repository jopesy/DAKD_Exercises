import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import itertools
import sklearn
from sklearn.neighbors import NearestNeighbors

csv_path = "winequality-red.csv"
data = pd.read_csv(csv_path, sep=';')

#Select input variables as x
x = data.iloc[0:,0:11]

#Select output variable (quality) as y
y = data.quality

#Returns the mean of the 'quality' values of the k neighbors of the given sample
def neighborsQualityMean(data,sampleIndex,k):
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(data)
    distances, indices = nbrs.kneighbors(data)
    
    splittedX = data[0:,0:11]
    splittedY = data[:,11]
    qualityValues = []
    
    for i in range(0, k):
        qualityValue = (splittedY[indices[sampleIndex][i]])
        qualityValues.append(qualityValue)
        
    nQualityValues = np.array(qualityValues)
    meanQuality = np.mean(nQualityValues)
    return meanQuality
    


def calculateOutOfSampleErrorVectors(dataIn, k):
    #Split whole data into 5 parts
    splittedData = np.array_split(dataIn, 5)

    outOfSampleErrorVectors = []

    for part in splittedData:
        splittedX = np.array(part.iloc[0:,0:11])
        splittedY = np.array(part.quality)
        errors = []    

        for i in range(0, len(splittedY)):        
            #calculate errors for each sample by subtracting the mean of the sample's
            #neighbors' quality values from the samples's quality value
            error = float(splittedY[i] - neighborsQualityMean(part, i, k))
            errors.append(error)

        outOfSampleErrorVectors.append(errors)
        
        #print(outOfSampleErrorVector)
                          
    return outOfSampleErrorVectors

def calculateInSampleErrorVectors(dataIn, k):
    splittedData = np.array_split(dataIn, 5)
    
    indexCombinations = list(itertools.combinations(range(0, len(splittedData)), 4))
    
    inSampleDataList = []    
    
    for combination in indexCombinations:
        inSampleData = []
        for index in combination:
            dataInIndex = list(np.array(splittedData[index]))
            inSampleData += dataInIndex
        inSampleData = np.array(inSampleData)
        inSampleDataList.append(inSampleData)
        
    inSampleErrorVectors = []
    
    for dataPart in inSampleDataList:
        splittedX = dataPart[0:,0:11]
        splittedY = dataPart[:,11]
        errors = []
        
        for i in range(0, len(splittedY)):
            #calculate errors for each sample by subtracting the mean of the sample's
            #neighbors' quality values from the samples's quality value
            error = float(splittedY[i] - neighborsQualityMean(dataPart, i, k))
            errors.append(error)

        inSampleErrorVectors.append(errors)
                          
    return inSampleErrorVectors

def concatenateVectors(vectors):
    vectorsConcatenated = []
    for vector in vectors:
        vectorsConcatenated += vector
    return vectorsConcatenated

def plotHistogram(concatVector, title):
    plt.figure()
    plt.hist(concatVector, bins=20, color="red")
    plt.title(title), plt.xlabel(""), plt.ylabel("Frequency")
    plt.show()

'''
#Out-of-sample errors of nearest neighbors
for k in range (1, 9):
    errorVectorsOut = calculateOutOfSampleErrorVectors(data, k)
    concatenatedVectorOut = np.array(concatenateVectors(errorVectorsOut))
    print(np.mean(concatenatedVectorOut))
    #plotHistogram(concatenatedVectorOut, "Out-of-sample error histogram, k="+str(k))
'''
    
for k in range (1, 9):
    errorVectorsIn = calculateInSampleErrorVectors(data, k)
    concatenatedVectorIn = np.array(concatenateVectors(errorVectorsIn))
    print(np.mean(concatenatedVectorIn))
    #plotHistogram(concatenatedVectorOut, "Out-of-sample error histogram, k="+str(k))


    