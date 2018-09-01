
# coding: utf-8

# In[109]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from pandas import read_csv
from sklearn.preprocessing import Imputer
import re
import seaborn as sns
import matplotlib.pyplot as plt


# In[96]:


train = pd.read_csv('/mnt/c/Users/sahib/Documents/Titanic Project/train.csv')
test = pd.read_csv('/mnt/c/Users/sahib/Documents/Titanic Project/test.csv')
gender_submission = pd.read_csv('/mnt/c/Users/sahib/Documents/Titanic Project/gender_submission.csv')


# In[5]:


train.describe()


# In[6]:


test.describe()


# In[7]:


# both train and test data have similar distributions which is good
# join data sets to create same manipulations in both


# In[68]:


# check missing values
train.isnull().sum()


# In[73]:


train['Age']


# In[97]:


# impute missing values for Age in train data with mean Age values
train['Age'] =  train['Age'].fillna(train['Age'].mean())

# impute missing values in Embarked with S
train['Embarked'] =  train['Embarked'].fillna('S')

train.isnull().sum()


# In[84]:


train.dtypes


