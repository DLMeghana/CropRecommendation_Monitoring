#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import print_function
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn import tree
import warnings
warnings.filterwarnings('ignore')


# In[4]:


PATH = 'D:\Final_Project\CropRecommendation_Monitoring\Crop_recommendation\Crop_recommendation.csv'
df = pd.read_csv(PATH)


# In[5]:


df.head()


# In[6]:


df.tail()


# In[7]:


df.size


# In[8]:


df.shape


# In[9]:


df.columns


# In[10]:


df['label'].unique()


# In[11]:


df.dtypes


# In[12]:


df.info()


# In[13]:


df.isnull().sum()


# In[14]:


df.duplicated().sum()


# In[15]:


df.describe()


# In[16]:


df['label'].value_counts()


# In[17]:


df.head()


# In[18]:


df_new = df.copy()

df_new.drop('label', axis=1, inplace=True)


# In[19]:


sns.heatmap(df_new.corr(),annot=True)


# In[20]:


features = df[['N', 'P','K','temperature', 'humidity', 'ph', 'rainfall']]
target = df['label']
labels = df['label']


# In[21]:


# Initializing empty lists to append all model's name and corresponding name
acc = []
model = []


# In[22]:


# Splitting into train and test data

from sklearn.model_selection import train_test_split
Xtrain, Xtest, Ytrain, Ytest = train_test_split(features,target,test_size = 0.2,random_state =2)


# In[23]:


from sklearn.naive_bayes import GaussianNB

NaiveBayes = GaussianNB()

NaiveBayes.fit(Xtrain,Ytrain)

predicted_values = NaiveBayes.predict(Xtest)
x = metrics.accuracy_score(Ytest, predicted_values)
acc.append(x)
model.append('Naive Bayes')
print("Naive Bayes's Accuracy is: ", x)

print(classification_report(Ytest,predicted_values))


# In[24]:


from sklearn.model_selection import cross_val_score


# In[25]:


# Cross validation score (NaiveBayes)
score = cross_val_score(NaiveBayes,features,target,cv=5)
score


# In[26]:


import pickle
# Dump the trained Naive Bayes classifier with Pickle
NB_pkl_filename = 'NBClassifier.pkl'
# Open the file to save as pkl file
NB_Model_pkl = open(NB_pkl_filename, 'wb')
pickle.dump(NaiveBayes, NB_Model_pkl)
# Close the pickle instances
NB_Model_pkl.close()


# In[27]:


from sklearn.svm import SVC

SVM = SVC(gamma='auto')

SVM.fit(Xtrain,Ytrain)

predicted_values = SVM.predict(Xtest)

x = metrics.accuracy_score(Ytest, predicted_values)
acc.append(x)
model.append('SVM')
print("SVM's Accuracy is: ", x)

print(classification_report(Ytest,predicted_values))


# In[28]:


score = cross_val_score(SVM,features,target,cv=5)
score


# In[29]:


import pickle
# Dump the trained Naive Bayes classifier with Pickle
SVM_pkl_filename = 'SVMClassifier.pkl'
# Open the file to save as pkl file
SVM_Model_pkl = open(SVM_pkl_filename, 'wb')
pickle.dump(SVM, SVM_Model_pkl)
# Close the pickle instances
SVM_Model_pkl.close()


# In[30]:


from sklearn.linear_model import LogisticRegression

LogReg = LogisticRegression(random_state=2)

LogReg.fit(Xtrain,Ytrain)

predicted_values = LogReg.predict(Xtest)

x = metrics.accuracy_score(Ytest, predicted_values)
acc.append(x)
model.append('Logistic Regression')
print("Logistic Regression's Accuracy is: ", x)

print(classification_report(Ytest,predicted_values))


# In[31]:


# Cross validation score (Logistic Regression)
score = cross_val_score(LogReg,features,target,cv=5)
score


# In[32]:


import pickle
# Dump the trained Naive Bayes classifier with Pickle
LR_pkl_filename = 'LogisticRegression.pkl'
# Open the file to save as pkl file
LR_Model_pkl = open(LR_pkl_filename, 'wb')
pickle.dump(LogReg, LR_Model_pkl)
# Close the pickle instances
LR_Model_pkl.close()


# In[33]:


from sklearn.ensemble import GradientBoostingClassifier

GradBoost = GradientBoostingClassifier()

GradBoost.fit(Xtrain,Ytrain)

predicted_values = GradBoost.predict(Xtest)

x = metrics.accuracy_score(Ytest, predicted_values)
acc.append(x)
model.append('Gradient Boosting')
print("Gradient Boosting Accuracy is: ", x)

print(classification_report(Ytest,predicted_values))


# In[34]:


score = cross_val_score(GradBoost,features,target,cv=5)
score


# In[35]:


import pickle
# Dump the trained Naive Bayes classifier with Pickle
GB_pkl_filename = 'GradientBoosting.pkl'
# Open the file to save as pkl file
GB_Model_pkl = open(GB_pkl_filename, 'wb')
pickle.dump(GradBoost, GB_Model_pkl)
# Close the pickle instances
GB_Model_pkl.close()


# In[36]:


plt.figure(figsize=[10,5],dpi = 100)
plt.title('Accuracy Comparison')
plt.xlabel('Accuracy')
plt.ylabel('Algorithm')
sns.barplot(x = acc,y = model,palette='dark')


# In[37]:


GB = GradientBoostingClassifier()
GB.fit(Xtrain,Ytrain)

Ypred= GB.predict(Xtest)

metrics.accuracy_score(Ytest, Ypred)


# In[38]:


def recommendation(N,P,K,temperature,humidity,ph,rainfall):
  features=np.array([[N,P,K,temperature,humidity,ph,rainfall]])
  prediction=GB.predict(features).reshape(1,-1)

  return prediction[0]


# In[39]:


N=3
P=3
K=7
temperature=4
humidity=20
ph=30
rainfall=50
predict=recommendation(N,P,K,temperature,humidity,ph,rainfall)
crop_dict=["Rice","Maize","Jute","Cotton","Coconut","Papaya","Orange","Apple","Maskmelon","Watermelon","Grapes","Mango","Banana","Pomegranate","Lentil","Blackgram","Mungbean","Mothbeans","Pigeonpeas","Kidneybeans","Chickpea","Coffee"]
if predict[0].title() in crop_dict:
  crop=predict[0].title()
  print("{} is a best crop to be cultivated ".format(crop))
else:
  print("Sorry we are not able to recommend a proper crop for this environment")


# In[40]:


import pickle
pickle.dump(GB,open('model.pkl','wb'))


# In[ ]:




