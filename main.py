print("CS6840 HW 1")
import os
os.chdir('c:\\Users') # Replace with path for executing. (so many danged can't find file errors....!)
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
import matplotlib.pyplot as plt
## Written by Jacob Ross for CS4840 - Spring 2019
# stolen and branched by Daniel Ketterer, Travis Ruddy, Benjamin Zook
# 
#

# Demonstrate:
# The ability to forego sleep, food, friendships and meaning to complete
# all tasks assigned. MUST COMPUTE 001001010100

 

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

	XTX = np.dot(xFeatures.T,Xfeatures)
	XTY = np.dot(X.T,Y)
	parameters = np.dot(np.linalg.inv(XTX),XTY)
	print parameters

"""
***RANDOM INITIALIZATION***
Calculate the hypothesis = X * theta
Calculate the cost = (h - y)^2 
Calculate the gradient = sum(X' * loss )/ m  #this is the part that makes it batch
Update the parameters theta = theta - alpha * gradient
Check if cost is less than epsilon
"""
def bgd(xFeatures, Ylabels, alpha, epsilon, epochs)

	#Cost = sum([(theta_0 + theta_1*x[i] - y[i])**2 for i in range(m)])


"""
TODO
3. Test each algorithm on the reDataUCI dataset from Task 1. Compare the
parameters found in Task 1 versus the parameters found with your OLS
and BGD code. Compare the SSE and MSE of all three approaches.
What might explain the differences (if any)?

4. Report the affects of trying a variety of learning rates and number of
epochs. What seems to be a good learning rate and number of epochs
for this data set? Plot the cost of the bgd function after each epoch for
a variety of number of epochs and learning rate. For each plot note the
behavior of the cost function. Is it decreasing as expected? Note any
issues that may be diagnosed via these plots.

5. Repeat tasks 1.3, 2.3 and 2.4 for another dataset of your choosing. You
may use any dataset you wish from any public repository (UCI, Kaggle,
etc.). Give a brief description of the dataset (features, labels).

6. CS 6840 only - come up with a methodology for comparing the speed
of your OLS and BGD implementation as a function of the size of some
artificial training data (that you will have to create). Plot the speed of
each algorithm as a function of the number of samples in the training
data. You may also have to adjust the number of features to get noticable
differences between your OLS code and your BGD code.


"""


#References

#https://the-tarzan.com/2012/10/27/calculate-ols-regression-manually-in-python-using-numpy/
#https://machinelearningmastery.com/gradient-descent-for-machine-learning/
#https://towardsdatascience.com/difference-between-batch-gradient-descent-and-stochastic-gradient-descent-1187f1291aa1

