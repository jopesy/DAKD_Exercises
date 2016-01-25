import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import itertools
from sklearn.linear_model import LinearRegression 


csv_path = "winequality-red.csv"
data = pd.read_csv(csv_path, sep=';')


#Select input variables as x
x = data.iloc[0:,0:11]

#Select output variable (quality) as y
y = data.quality

#Print attribute values of first sample
#print(x.loc[0])

#Print fixed acidity value of first sample
#print(x.loc[0].get_value(0))

#Quality of first sample
#print(y.loc[0])

#print(data)

#Calculate coefficients using statsmodels
#lm = smf.ols(formula='quality ~ fixed acidity + chlorides', data=data).fit()
#print (lm.params)

#Calculate coefficients to get the weight vector for the whole data
lm = LinearRegression()
lm.fit(x, y)

#print(lm.intercept_)
#print(lm.coef_)

weightVector = np.hstack((np.array(lm.intercept_), np.array(lm.coef_)))
print("Weight vector for the whole data: ")
print (weightVector)

#Split input variable values into 5 parts
#splittedX = np.array_split(x, 5)

#Split output variable values into 5 parts
#splittedY = np.array_split(y, 5)

#Print 1st part
#print (splittedData[0])

def calculateOutOfSampleErrorVectors(dataIn):
    #Split whole data into 5 parts
    splittedData = np.array_split(dataIn, 5)

    outOfSampleErrorVector = []
    inSampleErrorList = []

    for part in splittedData:
        splittedX = np.array(part.iloc[0:,0:11])
        splittedY = np.array(part.quality)
        errors = []    

        for i in range(0, len(splittedY)):        
            sampleReshaped = splittedX[i].reshape(1, -1)
            #make the predictions for each sample
            predictedSample = lm.predict(sampleReshaped)
            #calculate errors for each sample by substracting the predictions from the quality value
            error = float(splittedY[i] - predictedSample)
            errors.append(error)

        outOfSampleErrorVector.append(errors)
        
        #print(outOfSampleErrorVector)
                          
    return outOfSampleErrorVector
        
def calculateInSampleErrorVectors(dataIn):
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
        
    inSampleErrorVector = []
    
    for dataPart in inSampleDataList:
        splittedX = dataPart[0:,0:11]
        splittedY = dataPart[:,11]
        #subtract part from the whole data set
        errors = []
        
        for i in range(0, len(splittedY)):        
            sampleReshaped = splittedX[i].reshape(1, -1)
            #make the predictions for each sample
            predictedSample = lm.predict(sampleReshaped)
            #calculate errors for each sample by substracting the predictions from the quality value
            error = float(splittedY[i] - predictedSample)
            errors.append(error)

        inSampleErrorVector.append(errors)
        
        #print("*********")
        #print(inSampleErrorVector)
                          
    return inSampleErrorVector
    
                      
def concatenateVectors(vectors):
    vectorsConcatenated = []
    for vector in vectors:
        vectorsConcatenated += vector
    return vectorsConcatenated

def plotHistogram(concatVector, title):
    plt.figure()
    plt.hist(concatVector, bins=60, color="red")
    plt.title(title), plt.xlabel(""), plt.ylabel("Frequency")
    plt.show()
              
    
outOfSampleErrorVector = calculateOutOfSampleErrorVectors(data)
inSampleErrorVector = calculateInSampleErrorVectors(data)
outOfSampleErrorVector = calculateOutOfSampleErrorVectors(data)
inSampleErrorVector = calculateInSampleErrorVectors(data)
concatenatedOutSample = concatenateVectors(outOfSampleErrorVector)
concatenatedInSample = concatenateVectors(inSampleErrorVector)

print(len(concatenatedOutSample))
print(len(concatenatedInSample))

plotHistogram(concatenatedOutSample, "Out-of-sample error histogram")
plotHistogram(concatenatedInSample, "In-sample error histogram")

numpyOut = np.array(concatenatedOutSample)
meanOut = np.mean(numpyOut)
print("Mean of Out of sample errors: ")
print (meanOut)
       
numpyIn = np.array(concatenatedInSample)
meanIn = np.mean(numpyIn)
print("Mean of in-sample errors: ")
print (meanIn)

varIn = np.var(numpyIn)
varOut = np.var(numpyOut)

print("Variance of in-sample error")
print(varIn)
print("Variance of out-of-sample error")
print(varOut)
        
        
    
