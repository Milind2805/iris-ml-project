import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
dataset=pd.read_csv(r"C:\Users\mks90\OneDrive\Desktop\Iris.csv")
print(dataset.head(3))
print(dataset.isnull())
print(dataset.isnull().sum())
print(dataset.describe())
sns.pairplot(data=dataset,hue="Species")
plt.show()
x=dataset.iloc[:,:-1]
y=dataset["Species"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
x_train=ss.fit_transform(x_train)
x_test=ss.transform(x_test)
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(x_train,y_train)

print(lr.score(x_test,y_test)*100)
from sklearn.metrics import confusion_matrix,precision_score,recall_score,f1_score
cf=confusion_matrix(y_test,lr.predict(x_test))
sns.heatmap(cf,annot=True)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(x_train, y_train)
print("Naive Bayes Accuracy:", nb.score(x_test, y_test) * 100)
print("Logistic Regression Accuracy:", lr.score(x_test, y_test) * 100)

import numpy as np
new_flower = ss.transform([[5.1, 3.5, 1.4, 0.2]])
prediction = lr.predict(new_flower)
print("Predicted Species:", prediction[0])
                    
