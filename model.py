import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler


df = pd.read_csv('Loan_approvals.csv')
# print(df.head())

# print(df.info())

# print(df.isnull().sum())

data = df.dropna()
# print(data.isnull().sum())

data.replace({"Loan_Status":{'N':0,'Y':1}},inplace=True)
# print(data.head())

# print(data['Dependents'].value_counts())
data['Dependents'].replace({'3+':4},inplace=True)
# print(data['Dependents'].value_counts())


# convert categorical columns to numerical values-----------------
data.replace({'Married':{'No':0,'Yes':1},'Gender':{'Male':1,'Female':0},'Self_Employed':{'No':0,'Yes':1},'Property_Area':{'Rural':0,'Semiurban':1,'Urban':2},'Education':{'Graduate':1,'Not Graduate':0}},inplace=True)
# print(data)


# --------------------------------------Model-------------------------

label_encoder = preprocessing.LabelEncoder()

data['Gender'] = label_encoder.fit_transform(data['Gender'])
data['Married'] = label_encoder.fit_transform(data['Married'])
data['Education'] = label_encoder.fit_transform(data['Education'])
data['Self_Employed'] = label_encoder.fit_transform(data['Self_Employed'])
data['Property_Area'] = label_encoder.fit_transform(data['Property_Area'])
data['Loan_Status'] = label_encoder.fit_transform(data['Loan_Status'])

X = data[['Gender', 'Education', 'ApplicantIncome', 'Credit_History', 'Property_Area'  ]]
Y = data['Loan_Status']

# print(X)
# print(Y)

# Split 
X_train, X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.1,stratify=Y,random_state=2)
# print(X.shape, X_train.shape, X_test.shape)


# Model


sc = StandardScaler()


X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)



model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train,Y_train)

# Y_pred = model.predict(X_test)

# print(Y_pred)


# print(classification_report(Y_test,Y_pred))
# print(confusion_matrix(Y_test,Y_pred))



print("Prediction is : ", end="")

# Example model prediction
array = model.predict([[1, 0, 00, 0, 0]])

# Access the first element of the array (since it's a 2D array)
result = array[0]

# Print the result (for debugging or checking)
print(result, end=" ")

# Check if the result is 1 and print the appropriate response
if result == 1:
    print("Yes")
else:
    print("No")
