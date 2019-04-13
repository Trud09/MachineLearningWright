import numpy as np
#import matplotlib.pyplot as plt ### uncomment this
numCols = numRows = 5
#import matplotlib.pyplot as plt
def bgd(xFeatures, yLabels, alpha, epsilon, epochs):
    i = 0
    theta = np.array([[0 for x in range(numCols-1)]], ndmin=2)
   # print(theta)
   # theta.shape = (numCols-1,1)
    theta = np.transpose(theta)
    print(theta)
    Cost = epsilon + 1
    while i < epochs or Cost < epsilon:
        Hypo = np.dot(xFeatures, theta)
        Diff = Hypo - yLabels
       # print(Diff)
       # print(Hypo)
       # print(yLabels)
        Cost = (1/2*numRows) * np.sum(np.square(Diff) ) 
       # print(Cost)
        theta = theta - alpha * (1.0/numRows) * np.dot(np.transpose(xFeatures), Diff)
        #print(theta)
        i += 1
        #print(i)
    return theta
x = y = [9,8,7,6]
xFeatures = yLabels = x
#test = bgd(xFeatures, yLabels, .0000001, .0000001, 1000)

#test2 = ordinaryLeastSquares(xFeatures, yLabels)

CostHistory = []
#Here is where a variety of alpha, epochs are tested
for i in range(2):
 alpha = 1000**(-i-1)
 for j in range(5):
  epochs =10**(j)
  theta = bgd(xFeatures, yLabels, alpha, .0000001, epochs)
  Hypo = np.dot(xFeatures, theta)
  sse = np.sum(np.square(np.subtract(yLabels,Hypo)))
  mse = np.mean(np.square(np.subtract(yLabels,Hypo)))
  print('SSE and MME: alpha and epochs ' + str(sse) + str(', ') + str(mse) + str(', ') + str(alpha) + str(', ') + str(epochs))
  Diff = Hypo - yLabels
  Cost = (1/2*numRows) * np.sum(np.square(Diff) )
  CostHistory.append(Cost)
 fig = plt.figure()
 plt.plot(epochs, Cost, color = 'r')
 fig.suptitle("alpha = " + str(alpha))
 plt.xlabel("Epoch #")
 plt.ylabel("Cost")
 plt.show()
