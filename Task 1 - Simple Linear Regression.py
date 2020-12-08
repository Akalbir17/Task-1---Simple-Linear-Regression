"""
@author: Akalbir
"""
#importing dependencies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing dataset
df = pd.read_csv('task1_data.csv')
df.head()
print('--------Dataset Imported Sucessfully!---------')

#Plotting data for Visualization 
plt.scatter(df.Hours,df.Scores, color = 'blue')
plt.title('Hours V/S Percentage')
plt.xlabel('Hours of Study')
plt.ylabel('Score of Students')
plt.show()
print('-------Data Visualized Sucessfully!---------')

#Changing data attributes as input and output
x = df.iloc[:,:-1 ].values
y = df.iloc[:, 1].values

from sklearn.model_selection import train_test_split
#Splitting data for training and testing
x_train, x_test,y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state=0)


from sklearn import linear_model
#Training the data for Linear Regression
regr = linear_model.LinearRegression()
regr.fit(x_train,y_train)
print('----------Training Complete-----------')
print('Coefficients :',regr.coef_)
print('Intercepts:',regr.intercept_)


#Plotting the regression line
line = regr.coef_*x + regr.intercept_

#Plotting for the test data
plt.scatter(x,y)
plt.plot(x,line);
plt.show()

#Prediction of the scores
y_pred = regr.predict(x_test)


# Comparing Predicted and Actual Values
df_pred = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
print(df_pred)

#Predicting the score for 9.25 hours
hour = np.array([[9.25]])
h_pred = regr.predict(hour)
print('The score for {} hours is :{}'.format(hour,h_pred[0]))

