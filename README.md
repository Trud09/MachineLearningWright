# MachineLearningWright
This is a repository to hold our python project 1 in Wright State University CS6840
Project Outline

Task 1
1
•  Fit a linear regression model to the data without any feature transformsa. 
a. I/O data input 
 i. Data = np.loadtxt('CS6840DATA1.csv', delimiter = ",", skiprows = 1)#should be same syntax
 from linear regression example
2
a. Report the model parameters. What is the SSE and MSE?
3. After fitting the model, check the assumptions we must make with linear
regression. 
• Linear relationship between each variable and the labels.
• Residual independence
• Homoscedasticity of the residuals
• Co-linearity of the features.
a. With 4 indep and 1 depend var, we will need 4 plots x ( variable label plot, variable residual 
plot )
b. function for Pearson Coeff

4. Apply any technique you see fit to try and reduce the MSE and SSE
a.The simplest mechanism would be to make polynomial functions of the features. given features x1,
x2 we can make x3 = f( x1, x2) = x1^2 + x2^2 etc and see how well x3 fits
5. Report how your modifications to the model/features impact the assumptions you checked in (3).
6. Iterate between tasks (4) and (5) and try and reduce the SSE further.

Task 2

1. "
Write a Python function that takes a matrix of data trainingData and
a vector of labels labels and returns a vector of the optimal parameters
(parameters) via Ordinary Least Squares:
def ordinaryLeastSquares(trainingData, labels):
# your code here

return parameters
"
2. "
Write a Python function that takes a matrix of training data trainingData,
a vector of labels labels, a learning rate alpha, a stopping criterion
epsilon, and a max number of epochs epochs and returns a vector of
the optimal parameters (parameters) via the Batch Gradient Descent
Algorithm:
def bgd(trainingData, labels, alpha, epsilon, epochs)
# your code here
return parameters
"
3."
Test each algorithm on the reDataUCI dataset from Task 1. Compare the
parameters found in Task 1 versus the parameters found with your OLS
and BGD code. Compare the SSE and MSE of all three approaches.
What might explain the differences (if any)?
"
4. Report the affects of trying a variety of learning rates and number of
epochs. What seems to be a good learning rate and number of epochs
for this data set? Plot the cost of the bgd function after each epoch for
a variety of number of epochs and learning rate. For each plot note the
behavior of the cost function. Is it decreasing as expected? Note any
issues that may be diagnosed via these plots.

make alpha and epochs each a variable and create a for loop in which alpha
changes each iteration, inside that another one iterating epochs. 
let n range from -5 to 5 and set alpha = 10 ^ n

5. Repeat tasks 1.3, 2.3 and 2.4 for another dataset of your choosing. You
may use any dataset you wish from any public repository (UCI, Kaggle,
etc.). Give a brief description the dataset (features, labels).
6. CS 6840 only
Create time plots as a function of sample size 
