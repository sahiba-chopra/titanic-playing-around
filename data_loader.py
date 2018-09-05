
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



def load_data(data_path, pickle_path):
    """Loads csv data
    
    Parameter
    ---------
    path : Path where csv is located
    
    Returns
    -------
    data : DataFrame containing the csv data
    """
    if path.exists(pickle_path):
        data = pd.read_pickle(pickle_path)
    else:
        data = pd.read_csv(data_path)
        data.to_pickle(pickle_path)
    return data


