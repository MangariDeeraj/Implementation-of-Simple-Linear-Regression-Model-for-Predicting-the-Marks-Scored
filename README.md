# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn.
4.Assign the points for representing in the graph. 5.Predict the regression for marks by using the representation of the graph. 6.Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by:MANGARI DEERAJ
RegisterNumber:212223100031
*/
```

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
df.head()
df.tail()
x = df.iloc[:,:-1].values
x
y = df.iloc[:,1].values
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
y_pred
y_test
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='purple')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
 ``` 
## Output:

df.head()

![image](https://github.com/user-attachments/assets/04c564b7-1633-44f0-a9a3-9498e5d639b1)

df.tail()

![image](https://github.com/user-attachments/assets/4daeeaf5-ed1b-4dfb-afe2-1c4713539c62)

 Array value of X
 
![image](https://github.com/user-attachments/assets/265eea7f-a305-4578-b4eb-e8daba93fe1b)

Array value of Y

![image](https://github.com/user-attachments/assets/1a4bac8e-fac9-43a3-a587-aa97c6794b60)

Values of Y prediction

![image](https://github.com/user-attachments/assets/9dab98b7-8e89-43fc-ba96-dd6909c8dcfb)

Array values of Y test

![image](https://github.com/user-attachments/assets/f68f7a79-160e-4d3e-b9a9-06d241593eaa)

Training Set Graph

![image](https://github.com/user-attachments/assets/e1743835-5b5c-4d72-b381-aa58b793d7f0)

Test Set Graph

![image](https://github.com/user-attachments/assets/f84ca9e4-f5d4-44bd-a34b-511a697abb34)

Values of MSE, MAE and RMSE

![image](https://github.com/user-attachments/assets/fb02fc7d-15df-42c0-83f7-8d5caf47687a)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
