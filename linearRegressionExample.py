import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
import matplotlib.pyplot as plt
## Written by Jacob Ross for CS4840 - Spring 2019
## This is an exmaple of doing some basic things with Sci-kit learn.
## This code is not meant to be a shining example of good programming practices, rather
## is structured to demonstrate basic usage in the classroom.

## This code can be extended to do other forms of regression, combine features, etc.

## I demonstrate an improved linear regression model in terms of SSE by incorporating
## basic feature transforms. 


# Demonstrate:
#  - loading data from a file (CSV) (numpy)
#  - Manipulating, preprocessing data (numpy and sklearn)
#  - Linear Regression Assumption checking
#  - Fitting a model to loaded data (sklearn.linear_model)
#  - Showing fitted model attributes (SSE, coefficients, etc)
#  - Transforming features
#  - Showing model performance after 
 

#load data - skip first row which only contains metadata
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

#ID number
plt.figure()
plt.title('ID Number vs. Y')
plt.ylabel('Y label')
plt.xlabel('ID Number')
plt.scatter(xFeatures[:,0],yLabels)

# transaction date
plt.figure()
plt.title('Transaction Date vs. Y')
plt.ylabel('Y label')
plt.xlabel('Transaction Date')
plt.scatter(xFeatures[:,1],yLabels)

# house age
plt.figure()
plt.title('House Age vs. Y')
plt.ylabel('Y label')
plt.xlabel('House Age (years)')
plt.scatter(xFeatures[:,2],yLabels)

# distance to nearest MRT station
plt.figure()
plt.title('Dist to MRT vs. Y')
plt.ylabel('Y label')
plt.xlabel('Dist to MRT (meters)')
plt.scatter(xFeatures[:,3],yLabels)

# number of convenience stores
plt.figure()
plt.title('Number of convenience stores vs. Y')
plt.ylabel('Y label')
plt.xlabel('Num Stores (int)')
plt.scatter(xFeatures[:,4],yLabels)


# lat coord
plt.figure()
plt.title('Lat Coord vs. Y')
plt.ylabel('Y label')
plt.xlabel('Num Stores (int)')
plt.scatter(xFeatures[:,5],yLabels)

#long coord
plt.figure()
plt.title('Long coord vs. Y')
plt.ylabel('Y label')
plt.xlabel('Long Coord (double)')
plt.scatter(xFeatures[:,6], yLabels)

plt.show()






reg = LinearRegression()

# Find the best fit linear regression model
reg = reg.fit(xFeatures, yLabels)

# Predict new values based on some given samples.
# In this case, this fits the instances I used to create the model
# so I can do residual analysis
yPredicted  = reg.predict(xFeatures)

# The sum of square error: (yPredicted - yLabels)^2
# This should be the same as reg.residues_
sse = np.sum(np.square(np.subtract(yLabels,yPredicted)))

print('All features coefficients: ' + str(reg.coef_))
print('Intercept: ' + str(reg.intercept_))
print('reg.residue all features: '+ str(reg._residues))
print('SSE all features: '+ str(sse))
print('MSE all feautre: ' + str(np.mean(np.square(np.subtract(yLabels,yPredicted)))))

# Univariate Linear Regression with each feature individually
# The double bracket notation because it expects a "PANDAS" data frame as input



#X0 = id number
reg = LinearRegression().fit(xFeatures[:,[0]], yLabels)
predictedY = reg.predict(xFeatures[:,[0]])

print("Coefficent for single variable:" +  str(reg.coef_))
print("SSE for single variable 0:" + str(reg._residues))
#plt.figure()
#plt.scatter(xFeatures[:,0], yLabels, color = 'g')
#plt.plot(xFeatures[:,0], predictedY, color = 'r')


#X1 = the transaction date (for example, 2013.250=2013 March, 2013.500=2013 June, etc.) 
reg = LinearRegression().fit(xFeatures[:,[1]], yLabels)
predictedY = reg.predict(xFeatures[:,[1]])

#print("Coefficent for single variable:" +  str(reg.coef_))
print("SSE for single variable 1: " + str(reg._residues))
#plt.figure()
#plt.scatter(xFeatures[:,1], yLabels, color = 'g')
#plt.plot(xFeatures[:,1], predictedY, color = 'r')

#X2 = the house age (unit: year) 
reg = LinearRegression().fit(xFeatures[:,[2]], yLabels)
predictedY = reg.predict(xFeatures[:,[2]])

#print("Coefficent for single variable:" +  str(reg.coef_))
print("SSE for single variable 2: " + str(reg._residues))
#plt.figure()
#plt.scatter(xFeatures[:,2], yLabels, color = 'g')
#plt.plot(xFeatures[:,2], predictedY, color = 'r')

#X3 = the distance to the nearest MRT station (unit: meter) 
reg = LinearRegression().fit(xFeatures[:,[3]], yLabels)
predictedY = reg.predict(xFeatures[:,[3]])

#print("Coefficent for single variable:" +  str(reg.coef_))
print("SSE for single variable 3: " + str(reg._residues))
#plt.figure()
#plt.scatter(xFeatures[:,3], yLabels, color = 'g')
#plt.plot(xFeatures[:,3], predictedY, color = 'r')

#X4 = the number of convenience stores in the living circle on foot (integer) 
reg = LinearRegression().fit(xFeatures[:,[4]], yLabels)
predictedY = reg.predict(xFeatures[:,[4]])

#print("Coefficent for single variable:" +  str(reg.coef_))
print("SSE for single variable 4: " + str(reg._residues))
#plt.figure()
#plt.scatter(xFeatures[:,4], yLabels, color = 'g')
#plt.plot(xFeatures[:,4], predictedY, color = 'r')

#X5=the geographic coordinate, latitude. (unit: degree) 
reg = LinearRegression().fit(xFeatures[:,[5]], yLabels)
predictedY = reg.predict(xFeatures[:,[5]])

#print("Coefficent for single variable:" +  str(reg.coef_))
print("SSE for single variable 5: " + str(reg._residues))
#plt.figure()
#plt.scatter(xFeatures[:,5], yLabels, color = 'g')
#plt.plot(xFeatures[:,5], predictedY, color = 'r')

#X6=the geographic coordinate, longitude. (unit: degree)
reg = LinearRegression().fit(xFeatures[:,[6]], yLabels)
predictedY = reg.predict(xFeatures[:,[6]])

#print("Coefficent for single variable:" +  str(reg.coef_))
print("SSE for single variable 6: " + str(reg._residues))
#plt.figure()
#plt.scatter(xFeatures[:,6], yLabels, color = 'g')
#plt.plot(xFeatures[:,6], predictedY, color = 'r')

#Check correlation coefficients
r = np.corrcoef(xFeatures, rowvar = False)
plt.imshow(r, cmap ='hot', interpolation='nearest')
plt.colorbar()
plt.show()





# cut the "bad" variables and see what happens
xFeaturesTrimmed = np.hstack((xFeatures[:,[1]], xFeatures[:,[2]],xFeatures[:,[3]],xFeatures[:,[4]], xFeatures[:,[5]],xFeatures[:,[6]])       )

reg = LinearRegression().fit(xFeaturesTrimmed, yLabels)
print("SSE for trimmed features: " + str(reg._residues))





# Playing with some feature transforms:
#X1 = the transaction date (for example, 2013.250=2013 March, 2013.500=2013 June, etc.) 
# take the mod 1 of each date to hack off the year
x1 = xFeatures[:,[1]]
x1 = x1 % 1
x1 = np.sqrt(x1)

reg = LinearRegression().fit(x1 , yLabels)
predictedY = reg.predict(x1)


# take the log of feature 1
x1 = xFeatures[:,[1]]
x1 = np.log(x1)
reg = LinearRegression().fit(x1 , yLabels)
predictedY = reg.predict(x1)



#print("Coefficent for single variable:" +  str(reg.coef_))
print("SSE for single variable1 (mod appraoch):" + str(reg._residues))
# plt.figure()
# plt.scatter(x1, yLabels, color = 'g')
# plt.plot(x1, predictedY, color = 'r')

# log scale 3rd feature
x3 = xFeatures[:,[3]]
x3 = np.log10(x3)

reg = LinearRegression().fit(x3 , yLabels)
predictedY = reg.predict(x3)

#print("Coefficent for single variable:" +  str(reg.coef_))
print("SSE for single variable 3 (log):" + str(reg._residues))
plt.figure()
plt.scatter(x3, yLabels, color = 'g')
plt.plot(x3, predictedY, color = 'r')


# sqrt 3rd feature
x3 = xFeatures[:,[3]]
x3 = np.sqrt(x3)

reg = LinearRegression().fit(x3 , yLabels)
predictedY = reg.predict(x3)

#print("Coefficent for single variable:" +  str(reg.coef_))
print("SSE for single variable 3 (sqrt):" + str(reg._residues))
plt.figure()
plt.scatter(x3, yLabels, color = 'g')
plt.plot(x3, predictedY, color = 'r')




# log 5th feature
x5 = xFeatures[:,[5]]
x5 = np.log10(x5)

reg = LinearRegression().fit(x5 , yLabels)
predictedY = reg.predict(x5)

#print("Coefficent for single variable:" +  str(reg.coef_))
print("SSE for single variable 5 (log):" + str(reg._residues))
plt.figure()
plt.scatter(x5, yLabels, color = 'g')
plt.plot(x5, predictedY, color = 'r')



# log 6th feature
x6 = xFeatures[:,[6]]
x6 = np.log10(x6)

reg = LinearRegression().fit(x6 , yLabels)
predictedY = reg.predict(x6)

#print("Coefficent for single variable:" +  str(reg.coef_))
print("SSE for single variable 6 (log):" + str(reg._residues))
plt.figure()
plt.scatter(x6, yLabels, color = 'g')
plt.plot(x6, predictedY, color = 'r')



# sqrt 6th feature
x6 = xFeatures[:,[6]]
x6 = np.sqrt(x6)

reg = LinearRegression().fit(x6 , yLabels)
predictedY = reg.predict(x6)

#print("Coefficent for single variable:" +  str(reg.coef_))
print("SSE for single variable 6 (sqrt):" + str(reg._residues))
plt.figure()
plt.scatter(x6, yLabels, color = 'g')
plt.plot(x6, predictedY, color = 'r')



# square 6th feature
x6 = xFeatures[:,[6]]
x6 = np.square(x6)

reg = LinearRegression().fit(x6 , yLabels)
predictedY = reg.predict(x6)

#print("Coefficent for single variable:" +  str(reg.coef_))
print("SSE for single variable 6 (square):" + str(reg._residues))
plt.figure()
plt.scatter(x6, yLabels, color = 'g')
plt.plot(x6, predictedY, color = 'r')


#Fit the model again with the updated features:
xFeatures[:,1] = np.log(xFeatures[:,1])
xFeatures[:,3] = np.log(xFeatures[:,3])
xFeatures[:,5] = np.log(xFeatures[:,5])
xFeatures[:,6] = np.log(xFeatures[:,6])

reg = LinearRegression().fit(xFeatures, yLabels)
yPredicted  = reg.predict(xFeatures)
print("SSE for updated features: " + str(reg._residues))
print('MSE all features: ' + str(np.mean(np.square(np.subtract(yLabels,yPredicted)))))





exit

























