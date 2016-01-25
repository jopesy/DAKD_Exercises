import csv #import python's cvs module
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas
from pandas.tools.plotting import parallel_coordinates
from matplotlib.mlab import PCA as mlabPCA
from sklearn import manifold
from sklearn.metrics import euclidean_distances
import scipy.stats.mstats as ms
from IPython.display import HTML

csv_path = "winequality-red.csv"
with open(csv_path) as csvfile:
    data = csv.reader(csvfile, delimiter=';')

    #init lists for input variables/attributes	
    fixedAcidity = []; volatileAcidity = []; citricAcid =[]
    residualSugar = []; chlorides = []; freeSulfurDioxide = []
    totalSulfurDioxide = []; density = []; pH = []
    sulphates = []; alcohol = []
    #init list for output variable/attribute 
    quality = []

    #read lines and add values to corresponding lists      
    rows = 0
    wholeData = []
    for row in data:
        if rows >= 1: #exclude first row, which has the attribute names 
            wholeData.append(row)
            fixedAcidity.append(float(row[0]))
            volatileAcidity.append(float(row[1]))
            citricAcid.append(float(row[2]))
            residualSugar.append(float(row[3]))
            chlorides.append(float(row[4]))
            freeSulfurDioxide.append(float(row[5]))
            totalSulfurDioxide.append(float(row[6]))
            density.append(float(row[7]))
            pH.append(float(row[8]))
            sulphates.append(float(row[9]))
            alcohol.append(float(row[10]))
            quality.append(float(row[11]))    
        rows +=1
    csvfile.close()
    
#number of rows, minus the first row, is also the number of records here (1599)
n = rows - 1

#meanX = np.mean(numpyArray[:,0]) #Mean of fixed acidity

def calcCovMatrix(data):
    covMatrix = np.cov(numpyArray.T)
    return covMatrix
    
#Normalized given data using Zscore standardization    
def normalizeDataZscore(data):
    normalizedData = ms.zscore(numpyArray)
    return normalizedData

#Turn our whole data into a numpy array
numpyArray = np.array(wholeData).astype(np.float)
#Normalize data
normalizedNumpyArray = normalizeDataZscore(numpyArray)

#Compute two component PCA of given data and plot
def PCA(numpyArray):
    #Calculate covariance matrix
    #covMatrix = calcCovMatrix(numpyArray)
    covMatrix = calcCovMatrix(numpyArray)
    
    #Get the eigenvectors and eigenvalues from the covariance matrix
    eigenValues, eigenVectors = np.linalg.eig(covMatrix)
    #Form tuples of eigenValue, eigenVector
    eigenPairs = [(np.abs(eigenValues[i]), eigenVectors[:,i]) for i in range(len(eigenValues))]

    #Sort pairs from highest to lowest    
    eigenPairs.sort()
    eigenPairs.reverse()

    '''
    #Print eigenvalues
    for i in eigenPairs:
        print(i[0])
    '''

    #Choose the two eigenvectors with largest eigenvalues
    firstEigenVector = eigenPairs[0][1].reshape(12,1)
    secondEigenVector = eigenPairs[1][1].reshape(12,1)

    #Form a matrix using those two eigenvectors
    eigenMatrix = np.hstack((firstEigenVector, secondEigenVector))

    #Use the matrix to transform samples onto the new subspace
    transformed = eigenMatrix.T.dot(numpyArray.T)

    #Plot
    plt.plot(transformed[0,0:], transformed[1,0:], 'o', markersize=7, color='blue')
    plt.title("Principal component analysis (without normalization)")
    plt.xlabel("component 1")
    plt.ylabel("component 2")
    #plt.plot(transformed[0,20:40], transformed[1,20:40], '^', markersize=7, color='red')
    plt.show()    

#Calculate interquartile range of given sample
def IQR(sample):
    #firstQuartile = np.percentile(list, 25, interpolation="lower")
    #thirdQuartile = np.percentile(list, 75, interpolation="higher")
    firstQuartile = sorted(sample)[int(len(sample)*.25)]
    thirdQuartile = sorted(sample)[int(len(sample)*.75)]
    IQR = thirdQuartile - firstQuartile
    return IQR

#Calculate number of bins using Sturge's rule
def calcBinsSturges(sampleSize):
    return math.ceil(math.log(sampleSize,2)+1)

#Calculate number of bins using the Square-root choice
def calcBinsSquareRoot(sampleSize):
    return math.ceil(math.sqrt(sampleSize))

#Calculate number of bins using Freedman-Diaconis' rule
def calcBinsFD(sample):
    sampleIQR = IQR(sample)
    binWidth = 2*(sampleIQR / (n**(1/3.0)))
    sampleMin = sorted(sample)[0]
    sampleMax = sorted(sample)[len(sample)-1]
    numberOfBins = math.ceil((sampleMax - sampleMin) / binWidth)
    return numberOfBins

print('IQR of fixed acidity: ' + str(IQR(fixedAcidity)))
print('Number of bins: '+ str(calcBinsFD(fixedAcidity)))

#Plot histogram of given data
def plotHistogram(list, numberOfBins, title, labelX, labelY, barColor):
    plt.figure()
    plt.hist(list, bins=numberOfBins, color=barColor)
    plt.title(title), plt.xlabel(labelX), plt.ylabel(labelY)
    plt.show()
    #plt.savefig("Histogram_"+labelX+"_"+labelY+"_sqrt.png")
    
def plotParallelCoordinates():
    data = pandas.read_csv(csv_path, sep=';')
    plt.figure()
    parallel_coordinates(data, 'quality')
    plt.show()
    
def plotScatterPlot(sample1, sample2, sample1Name, sample2Name):
    plt.figure()
    plt.scatter(sample1,sample2) #c is color ot the markers
    plt.title("Scatter plot of "+sample1Name+" and "+sample2Name)
    plt.xlabel(sample1Name), plt.ylabel(sample2Name)
    plt.show()
    
#Plot 2D MDS Scatter plot using Euclidean distance matrix
def plot2DMDSScatterPlot(data):
    
    data = data
    #Calculate Euclidean distance matrix
    #distances = euclidean_distances(data)
    
    # Multidimensional scaling
    #mds = manifold.MDS(n_components=2, dissimilarity="precomputed", n_jobs=1)
    mds = manifold.MDS(n_components=2, dissimilarity="euclidean", n_jobs=1)
    
    #Calculate coordinates for the new 2D space
    coordinates = mds.fit(data).embedding_
    
    #Plot
    plt.plot(coordinates[:,0], coordinates[:,1], 'o', markersize=7, color='blue')
    plt.title("2D MDS Scatter plot")
    plt.xlabel("coordinate 1"), plt.ylabel("coordinate 2")
    plt.show()
 
dataframe = pandas.read_csv(csv_path, sep=';')
#Tells about correlation and direction
corrTablePearson = dataframe.corr(method='pearson')


corrTableKendall = dataframe.corr(method='kendall')
    
binsSturges = calcBinsSturges(n)    #Sturge's rule = 12 bins
binsSqrt = calcBinsSquareRoot(n)    #Square-root choice = 40 bins
binsFD_fixedAcidity = calcBinsFD(fixedAcidity)  #Freedman-Diaconis' rule = 32 bins
binsFD_volatileAcidity = calcBinsFD(volatileAcidity)
binsFD_density = calcBinsFD(density)
binsFD_alcohol = calcBinsFD(alcohol)
binsFD_citricAcid = calcBinsFD(citricAcid)
binsFD_residualSugar = calcBinsFD(residualSugar)

#plotParallelCoordinates()

#plotScatterPlot(fixedAcidity, volatileAcidity, "Fixed acidity", "Volatile acidity")
#plotScatterPlot(alcohol, density, "Alcohol", "Density")
#plotScatterPlot(chlorides, density, "Chlorides","Density")
#plotScatterPlot(alcohol, quality, "Alcohol", "Quality")
#plotScatterPlot(residualSugar, totalSulfurDioxide, "Residual sugar", "Total sulfur dioxide")
#plotScatterPlot(sulphates, chlorides, "Sulphates", "Chlorides")


#plot2DMDSScatterPlot(wholeData)
#PCA(numpyArray)
#PCA(normalizedNumpyArray)


#Turn tables into html
#htmlPearson = HTML(corrTablePearson.to_html())
#htmlKendall = HTML(corrTableKendall.to_html())

#Write into html files
#htmlFilePearson = open('correlationPearson.html', 'w')
#htmlFilePearson.write(htmlPearson.data)
#htmlFilePearson.close()

#htmlFileKendall = open('correlationKendall.html', 'w')
#htmlFileKendall.write(htmlKendall.data)
#htmlFileKendall.close()

