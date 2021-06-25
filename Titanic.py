#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
sns.set()
from collections import Counter


# In[2]:


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
df_train_raw = pd.read_csv('train.csv')
df_train_raw.head()


# In[3]:


male = df_train_raw.loc[df_train_raw.Sex == "male"]['Survived']
female = df_train_raw.loc[df_train_raw.Sex == "female"]['Survived']

plt.bar(['Male','Female'],[sum(male)/len(male),sum(female)/len(female)])
plt.title('Percentage of survival by sex', fontsize=15)
plt.ylabel('Survival rate', fontsize = 15)
plt.ylim(0,1)
plt.show

#Women are more prone to survive the accident. Therefore, sex must be a strong feature


# In[59]:


sns.histplot(x='Age',data=df_train_raw, hue='Sex')


# In[60]:


sns.histplot(x='Age', data = df_train_raw, hue='Survived', legend=True)

#Kids and teens are more likely to survive. "Women and children first!!"?


# In[4]:


sns.barplot(x='Pclass', y='Survived', data=df_train_raw)

#Passengers on the first and second class are more likely to survive (class 1 and 2
#are destinated to people with high fortune? Resulting on a preference to get to the boats?)


# In[6]:


sns.barplot(x='SibSp',y='Survived',data=df_train_raw)


# In[7]:


sns.barplot(x='Pclass',y='Survived',hue='Embarked',data=df_train_raw)


# In[8]:


#copy of the original dataset
#mapping the Sex and Embarked values
#replacing the missing Ages with the dataset's mean
df_train = df_train_raw.copy()
df_train['Sex'] = df_train_raw['Sex'].map({'male':0,'female':1, 'nan':'nan'})
df_train['Embarked'] = df_train_raw['Embarked'].map({'S':0,'C':1,'Q':2, 'nan':'nan'})
df_train['Age'] = df_train['Age'].fillna(round(df_train['Age'].mean(),0))
df_train = df_train.dropna(0,subset=['Embarked'])

#dummies
df_train[['bin_Male',"bin_Female"]] = pd.get_dummies(df_train['Sex'])
df_train[['ticket_1','ticket_2','ticket_3']] = pd.get_dummies(df_train['Pclass'])

df_train = df_train.drop(['Cabin','Ticket','Name','Sex','Pclass'],1)

#normalize Fare
for i in range (df_train['Fare'].shape[0]):
    try:
        df_train.loc[i,'Fare'] = (df_train.loc[i,'Fare']-df_train['Fare'].min())/(df_train['Fare'].max()-df_train['Fare'].min())
    except:
        i = i + 1
        
#normalize Age
for i in range (df_train['Age'].shape[0]):
    try:
        df_train.loc[i,'Age'] = (df_train.loc[i,'Age']-df_train['Age'].min())/(df_train['Age'].max()-df_train['Age'].min())
    except:
        i = i + 1
        
df_train.head(15)


# ### Logistic Regression - Training the model

# In[9]:


#inputs definition
inputs = sm.add_constant(df_train[['Age','Fare','bin_Male','bin_Female','ticket_1','ticket_2','ticket_3']])
target = df_train['Survived']

#model
log_reg = sm.Logit(target,inputs)
result = log_reg.fit()
print(result.summary())

results = [1 if ponto >=0.5 else 0 for ponto in result.predict(inputs)]
VP = sum([1 if results[i] == 1 and np.array(df_train["Survived"])[i] == 1 else 0 for i, _ in enumerate(results)])
FP = sum([1 if results[i] == 1 and np.array(df_train["Survived"])[i] == 0 else 0 for i, _ in enumerate(results)])
VN = sum([1 if results[i] == 0 and np.array(df_train["Survived"])[i] == 0 else 0 for i, _ in enumerate(results)])
FN = sum([1 if results[i] == 0 and np.array(df_train["Survived"])[i] == 1 else 0 for i, _ in enumerate(results)])

print ("VP:", VP)
print ("VN:", VN)
print ("FP:", FP)
print ("FN:", FN)

print ("accuracy: ", (VP+VN)/(VP+VN+FP+FN))
print ("precision: ", VP/(VP+FP))
print ("recall: ", VP/(VP+FN))


# ### Logistic Regression - Testing the model

# In[10]:


#importing the test dataset
df_test_raw = pd.read_csv('test.csv')
df_test_raw.head(10)


# In[11]:


#copy of the original dataset
#mapping the Sex and Embarked values
#replacing the missing Ages with the dataset's mean
df_test = df_test_raw.copy()
df_prediction = pd.DataFrame()
df_prediction['Name'] = df_test['Name']

df_test['Sex'] = df_test_raw['Sex'].map({'male':0,'female':1, 'nan':'nan'})
df_test['Embarked'] = df_test_raw['Embarked'].map({'S':0,'C':1,'Q':2, 'nan':'nan'})
df_test['Age'] = df_test['Age'].fillna(round(df_test['Age'].mean(),0))
df_test = df_test.dropna(0,subset=['Embarked'])

#dummies
df_test[['bin_Male',"bin_Female"]] = pd.get_dummies(df_test['Sex'])
df_test[['ticket_1','ticket_2','ticket_3']] = pd.get_dummies(df_test['Pclass'])

df_test = df_test.drop(['Cabin','Ticket','Name','Sex','Pclass'],1)

#normalize Fare
for i in range (df_test['Fare'].shape[0]):
    try:
        df_test.loc[i,'Fare'] = (df_test.loc[i,'Fare']-df_test['Fare'].min())/(df_test['Fare'].max()-df_test['Fare'].min())
    except:
        i = i + 1
        
#normalize Age
for i in range (df_test['Age'].shape[0]):
    try:
        df_test.loc[i,'Age'] = (df_test.loc[i,'Age']-df_test['Age'].min())/(df_test['Age'].max()-df_test['Age'].min())
    except:
        i = i + 1
        
df_test.head(15)


# In[12]:


inputs_test = sm.add_constant(df_test[['Age','Fare','bin_Male','bin_Female','ticket_1','ticket_2','ticket_3']])

results_survived = result.predict(inputs_test)
results_survived = [1 if i>=0.5 else 0 for i in results_survived]
df_prediction['Prediction'] = results_survived
df_prediction['Prediction_Meaning'] = df_prediction['Prediction'].map({0:'Died',1:'Survived'})
print(df_prediction['Prediction'].count()-df_prediction['Prediction'].sum())
df_prediction.head(20)


# In[65]:


#df_submit = pd.DataFrame()
#df_submit['PassengerId'] = df_test['PassengerId']
#df_submit['Survived'] = df_prediction['Prediction']
#df_submit.to_csv('ArthurPereira_SubmissionFile.csv',index=False)

