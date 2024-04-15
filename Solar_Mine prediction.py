#!/usr/bin/env python
# coding: utf-8

# **Importing the Dependencies**

# In[5]:


import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score 


# **Data Collection and Data Processing**

# In[8]:


df =pd.read_csv(r"C:\Users\Gopi\Downloads\Copy of sonar data.csv" ,header=None)


# In[9]:


df


# In[10]:


df.head()


# In[13]:


# Number of rows and columns

df.shape


# In[14]:


#describe ---> statistical measures of the data 
df.describe()


# In[15]:


df[60].value_counts()


# In[26]:


df[60]


# **M --> Mine**
# **R -->Rock**

# In[29]:


df.groupby(60).mean()


# In[19]:


# separating data and Labels
x = df.drop(columns=60,axis=1)
y = df[60]


# In[20]:


print(x)
print(y)


# **Training and Test Data**

# In[21]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.1 ,stratify=y,random_state=1)

print(x.shape,x_train.shape,x_test.shape)


# **Model Training --> Logistic Regression**

# In[23]:


model = LogisticRegression()


# In[24]:


# Training the Logistic Regression model with training data

model.fit(x_train,y_train)


# **Model Evaluation**

# In[30]:


# accuracy on training data
x_train_prediction = model.predict(x_train)
training_data_accuracy = accuracy_score(x_train_prediction,y_train)


# In[31]:


print('Accuracy on training data : ',training_data_accuracy)


# In[32]:


#accuracy on test data 
x_test_prediction = model.predict(x_test)
test_data_accuracy = accuracy_score(x_test_prediction,y_test)


# In[34]:


print('Accuracy on testing data : ',test_data_accuracy)


# **Making a Predictive System**

# In[39]:


input_data = (0.0522,0.0437,0.0180,0.0292,0.0351,0.1171,0.1257,0.1178,0.1258,0.2529,0.2716,0.2374,0.1878,0.0983,0.0683,0.1503,0.1723,0.2339,0.1962,0.1395,0.3164,0.5888,0.7631,0.8473,0.9424,0.9986,0.9699,1.0000,0.8630,0.6979,0.7717,0.7305,0.5197,0.1786,0.1098,0.1446,0.1066,0.1440,0.1929,0.0325,0.1490,0.0328,0.0537,0.1309,0.0910,0.0757,0.1059,0.1005,0.0535,0.0235,0.0155,0.0160,0.0029,0.0051,0.0062,0.0089,0.0140,0.0138,0.0077,0.0031,M
)

#changing the input_data to a numpy array
input_data _as_numpy_array =np.asarray(input_data)

#reshape the np array as we are predicting for one instance 
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction =model.predict(input_data_reshaped)
print(prediction)

if (prediction[0]=='R'):
    print('The object is a Rock')
else:
    print('The object is a Mine')


# In[ ]:




