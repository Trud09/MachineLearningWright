print("CS6840 HW 1")
#import os
#os.chdir('c:\\Users') # Replace with path 
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
import matplotlib.pyplot as plt
 
#load data - skip first row 
reData = np.loadtxt('reDataUCI.csv', delimiter = ",", skiprows = 1)


numRows = np.size(reData,0)
numCols = np.size(reData,1)

#Columns 0-6 are features
#X0= id number
#X1=the transaction date (for example, 2013.250=2013 March, 2013.500=2013 June, etc.) 
#X2=the house age (unit: year) 
#X3=the distance to the nearest MRT station (unit: meter) 
#X4=the number of convenience stores in the living circle on foot (integer) 
#X5=the geographic coordinate, latitude. (unit: degree) 
#X6=the geographic coordinate, longitude. (unit: degree)

xFeatures = reData[:,0:numCols-1]

#Last column are the labels
yLabels = reData[:,7]

"""
OLS Derivation
# Ax = b 
# A'Ax = A'b
# x = inverse(A'A)*A'b
# Here x is a feature vector, A is xFeatures, and b is Ylabels.
"""
def ordinaryLeastSquares(xFeatures, Ylabels):

	XTX = np.dot(xFeatures.T,xFeatures)
	XTY = np.dot(xFeatures.T, Ylabels)
	OLS_params = np.dot(np.linalg.inv(XTX),XTY)
	print(OLS_params)

"""
Calculate the hypothesis = X * theta
Calculate the cost = (h - y)^2 
Calculate the gradient = sum(X' * loss )/ m  #this is the part that makes it batch
Update the parameters theta = theta - alpha * gradient of cost 
Check if cost is less than epsilon
"""
def bgd(xFeatures, yLabels, alpha, epsilon, epochs):
    i = 0
    theta = np.array([[0 for x in range(numCols-1)]], ndmin=2)
   # print(theta)
    #theta.shape = (numCols-1,1)
    theta = np.transpose(theta)
    print(theta)
    Cost = epsilon + 1
    while i < epochs or Cost < epsilon:
        Hypo = np.dot(xFeatures, theta)
        Diff = Hypo - yLabels
       # print(Diff)
        print(Hypo)
       # print(yLabels)
        Cost = (1/2*numRows) * np.sum(np.square(Diff) ) 
        print(Cost)
        theta = theta - alpha * (1.0/numRows) * np.dot(np.transpose(xFeatures), Diff)
        print(theta)
        i += 1
        print(i)
    return theta
    
    
"""
TODO
3. Test each algorithm on the reDataUCI dataset from Task 1. Report the best one
4. Report the affects of trying a variety of learning rates and number of
epochs. 
Plot the cost of the bgd function after each epoch for
a variety of number of epochs and learning rate. 
"""
test = bgd(xFeatures, yLabels, .0000001, .0000001, 10000)

test2 = ordinaryLeastSquares(xFeatures, yLabels)

CostHistory = []
#Here is where a variety of alpha, epochs are tested
for i in range(5)
 alpha = 100^(-i)
 for j in range(5)
  epochs =100^(j)
  theta = bgd(xFeatures, yLabels, alpha, .0000001, epochs)
  Hypo = np.dot(xFeatures, theta)
  sse = np.sum(np.square(np.subtract(yLabels,Hypo)))
  mse = np.mean(np.square(np.subtract(yLabels,Hypo)))
  print('SSE and MME: alpha and epochs ' + str(sse) + str(mse) + str(alpha) + str(epochs))
  Diff = Hypo - yLabels
  Cost = (1/2*numRows) * np.sum(np.square(Diff) )
  CostHistory.append(Cost)
 fig = plt.figure()
 plt.plot(epochs, Cost, color = 'r')
 fig.suptitle("alpha = " + str(alpha))
 plt.xlabel("Epoch #")
 plt.ylabel("Cost")
 plt.show()
"""
5. Repeat tasks 1.3, 2.3 and 2.4 for another dataset of your choosing. You
may use any dataset you wish from any public repository (UCI, Kaggle,
etc.). Give a brief description of the dataset (features, labels).
"""

#this is using the new dataset
reData = np.loadtxt('Admission_Predict.csv', delimiter = ",", skiprows = 1)


numRows = np.size(reData,0)
numCols = np.size(reData,1)


xFeatures = reData[:,0:numCols-1]
yLabels = reData[:,numRows-1]

newtest = bgd(xFeatures, yLabels, .0000001, .0000001, 10000)

newtest2 = ordinaryLeastSquares(xFeatures, yLabels)



"""


#References

#https://the-tarzan.com/2012/10/27/calculate-ols-regression-manually-in-python-using-numpy/
#https://machinelearningmastery.com/gradient-descent-for-machine-learning/
#https://towardsdatascience.com/difference-between-batch-gradient-descent-and-stochastic-gradient-descent-1187f1291aa1
#https://www.kaggle.com/mohansacharya/graduate-admissions/version/2    ---Dataset
"""
