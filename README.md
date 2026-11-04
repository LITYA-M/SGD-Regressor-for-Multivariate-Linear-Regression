# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:

To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:

1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:

1. STEP 1: Import the required libraries and create a dataset containing CGPA, Aptitude Score, Communication Skill and Placement status.
2. STEP 2: Split the dataset into input features (X) and target variable (y), then divide it into training and testing data.
3. STEP 3: Train the Logistic Regression model using the training data and predict the placement results.
4. STEP 4: Evaluate the model using Accuracy Score, Confusion Matrix and Classification Report, then predict whether a new student will be placed or not.
   

## Program:

```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: LITYA M
RegisterNumber: 212225230152 
*/
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# Suppose we have: CGPA, Aptitude Score, Communication Skill → Placement (1=Placed, 0=Not Placed)
data = {
    'CGPA': [8.5, 6.8, 7.9, 5.4, 9.1, 8.0, 7.5, 6.0, 9.3, 5.8],
    'Aptitude_Score': [82, 55, 75, 48, 92, 77, 73, 50, 95, 45],
    'Communication_Skill': [8, 6, 7, 5, 9, 8, 7, 6, 9, 5],
    'Placed': [1, 0, 1, 0, 1, 1, 1, 0, 1, 0]
}

df = pd.DataFrame(data)
print("Dataset:\n", df, "\n")

X = df[['CGPA', 'Aptitude_Score', 'Communication_Skill']]
y = df['Placed']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)


print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
new_student = np.array([[8.2, 78, 8]])  # CGPA, Aptitude, Communication
prediction = model.predict(new_student)

print("\nNew Student Prediction:")
if prediction[0] == 1:
    print(" The student is likely to be PLACED.")
else:
    print(" The student is NOT likely to be placed.")
```

## Output:

<img width="753" height="667" alt="image" src="https://github.com/user-attachments/assets/03dad6a6-d188-4e01-b897-dac29951e096" />


## Result:

Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD 
regressor is written and verified using python programming.

..




..






..








..









..










..







..








..

..
..





..




..






..



..




..


..

...



..


..

..

...



..

...



..


..


...



..


..


..

..


..



..



..





..



..




..









...





..




..







..




...












...











..
