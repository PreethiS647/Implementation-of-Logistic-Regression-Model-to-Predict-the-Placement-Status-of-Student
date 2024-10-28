# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

STEP 1:Start

STEP 2:Load and preprocess the dataset: drop irrelevant columns, handle missing values, and encode categorical variables using LabelEncoder.

STEP 3:Split the data into training and test sets using train_test_split.

STEP 4:Create and fit a logistic regression model to the training data.

STEP 5:Predict the target variable on the test set and evaluate performance using accuracy, confusion matrix, and classification report.

STEP 6:Display the confusion matrix using metrics.ConfusionMatrixDisplay and plot the results.

STEP 7:End

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Preethi S
RegisterNumber: 212223230157

import pandas as pd
data=pd.read_csv('/content/Placement_Data (1).csv')
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1) #removes the specified row or column.
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
data1


x=data1.iloc[:,:-1]
x

y=data1['status']
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size =0.2,random_sta

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver='liblinear')# A library for large linear classification
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred


from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)


lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
*/
```

## Output:

# Dataset

![Screenshot 2024-09-14 160445](https://github.com/user-attachments/assets/058681b1-bb75-412f-b01d-701362346bdc)

# Head

![Screenshot 2024-09-14 160501](https://github.com/user-attachments/assets/d68d5c04-06c5-4e8f-8216-adc1c28eb18b)

# Null values

![Screenshot 2024-09-14 160509](https://github.com/user-attachments/assets/e436b60c-e51a-4132-8081-de1769faefae)

# Sum of Duplicate values

![Screenshot 2024-09-14 160517](https://github.com/user-attachments/assets/7373e66c-3351-4a2b-922b-bc3a345440d4)

# X and Y data

![Screenshot 2024-09-14 160542](https://github.com/user-attachments/assets/28bd8242-7e7c-4b64-b666-ac9e8346a8b0)

![Screenshot 2024-09-14 160555](https://github.com/user-attachments/assets/c278e919-56fd-468f-9dac-5ced028d5dd0)

# Train data

![Screenshot 2024-09-14 160604](https://github.com/user-attachments/assets/163ae553-8d6a-4389-84e5-c11e27aab4e0)

# Test data

![Screenshot 2024-09-14 160612](https://github.com/user-attachments/assets/eb53d7db-ad1e-4ac1-a6be-c8a667447d7f)

# Accuracy

![Screenshot 2024-09-14 160618](https://github.com/user-attachments/assets/b9bec65b-d38f-4c2c-8b9e-e65ab8dc8e38)

# Confusion Matrix

![Screenshot 2024-10-19 171311](https://github.com/user-attachments/assets/9cf68af0-674a-4067-96f2-fc83fd61a945)

# Precision

![Screenshot 2024-10-19 171322](https://github.com/user-attachments/assets/d4be70ab-a87b-4eb1-a9d5-6726665b29b2)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
