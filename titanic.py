from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import datasets
from sklearn import metrics
from sklearn.feature_selection import RFE 
import seaborn as sns
import numpy as np
import pandas as pd 

train1 = pd.read_csv("titanic/train.csv")
#summary stats
train1.describe()

#data type of column
train1.dtypes
train1["Survived"].astype("category")

#coercing data type into categories. 
train1["Survived"] = train1["Survived"].astype("category")
train1["Pclass"] = train1["Pclass"].astype("category")
train1["Sex"] = train1["Sex"].astype("category")
train1["Cabin"].head()
train1["Cabin"].astype("category").head()
train1["Embarked"] = train1["Embarked"].astype("category")

#checking for NA values
train1["Pclass"].isnull().sum()
train1["Sex"].isnull().sum()
train1["Age"].isnull().sum()
train1["SibSp"].isnull().sum()
train1["Parch"].isnull().sum()
train1["Fare"].isnull().sum()
train1["Cabin"].isnull().sum()
train1["Embarked"].isnull().sum()

#groupby and count to see frequency of male and female. 
train1.groupby("Sex").count()


#replacing NA in age by mean of male and female. 
train1["Age"][train1["Sex"]=="male"].mean()
train1["Age"][train1["Sex"]=="female"].mean()
train1["Age"].replace((train1["Age"].isnull()) & (train1["Sex"]=="female"), train1["Age"][train1["Sex"]=="female"].mean())
train1["Age"] = train1["Age"].fillna(30.726644591611478)


#replacing cabin columns with N,A,B,...T
train1["Cabin"] = train1["Cabin"].replace(np.nan,"N")
mask = train1["Cabin"].str.startswith("A")
col = "Cabin"
train1.loc[mask,col] = "A"
mask = train1["Cabin"].str.startswith("B")
col = "Cabin"
train1.loc[mask,col] = "B"
mask = train1["Cabin"].str.startswith("C")
col = "Cabin"
train1.loc[mask,col] = "C"
mask = train1["Cabin"].str.startswith("D")
col = "Cabin"
train1.loc[mask,col] = "D"
mask = train1["Cabin"].str.startswith("E")
col = "Cabin"
train1.loc[mask,col] = "E"
mask = train1["Cabin"].str.startswith("F")
col = "Cabin"
train1.loc[mask,col] = "F"
mask = train1["Cabin"].str.startswith("G")
col = "Cabin"
train1.loc[mask,col] = "G"
train1["Cabin"].replace(np.nan,"N")


train1["Cabin"] = train1["Cabin"].astype("category")

#Creating new variable family 
train1["Fam"] = train1["SibSp"] + train1["Parch"] 

#Finding correlation  
train1.corr()
#Fam and sibsp/parch correlated. 

#Exploratory Analysis

#count of survived and not survived. 
train1["Survived"].value_counts()

#If the mean values when grouped by survived differ significantly. 
train1.groupby("Survived").mean()

#For Plotting 
plt.rc("font",size=14)
sns.set(style="white")
sns.set(style="whitegrid",color_codes=True)

#CrossTable to see the number of people survived and not survived in Pclass1, Pclass2 and Pclass3. 
pd.crosstab(train1.Pclass,train1.Survived)

#Making a bar plot. Useful to see if some features are important. 
pd.crosstab(train1.Pclass, train1.Survived).plot(kind="bar")
pd.crosstab(train1.Sex,train1.Survived).plot(kind="bar")
pd.crosstab(train1.Embarked,train1.Survived).plot(kind="bar")

#Recursive Feature Selection. 
# NOTE  : Need to convert all categorical variables into one hot encoding for logistic regression. 

data = train1
#create one hot encoding 
cat_vars=['Pclass','Sex','Cabin','Embarked']
for var in cat_vars:
    cat_list='var'+'_'+var
    cat_list = pd.get_dummies(data[var], prefix=var)
    data1=data.join(cat_list)
    data=data1
cat_vars=['Pclass','Sex','Cabin','Embarked']
data_vars=data.columns.values.tolist()
to_keep=[i for i in data_vars if i not in cat_vars]

data_final=data[to_keep]
data_final.columns.values

#Creating final dataset. 
data_final_vars=data_final.columns.values.tolist()
y2=['Survived']
X2=[i for i in data_final_vars if i not in ["Name","Survived","PassengerId","Ticket","Fam","Cabin_A","Cabin_B","Cabin_C","Cabin_D","Cabin_E","Cabin_F","Cabin_G","Cabin_T"]]

#Performing RFE 
logreg = LogisticRegression()
rfe = RFE(logreg, 10)
rfe = rfe.fit(data_final[X2], data_final[y2])
print(rfe.support_)
print(rfe.ranking_)

#Keeping final 10 important features. 
cols = ["SibSp","Pclass_1","Pclass_2","Pclass_3","Sex_female","Sex_male","Cabin_N","Embarked_C","Embarked_Q","Embarked_S"]
X = data_final[cols]
y = data_final["Survived"]

#Splitting the data to train and test. 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

#fitting logistic regression. 
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))

#Final Accuracy = 0.81. 

