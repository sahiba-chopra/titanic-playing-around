
# coding: utf-8

# In[4]:


from utils import *


# In[6]:


from data_loader import *


# In[8]:


train.head()


# In[9]:


train.shape


# In[ ]:


# pearson correlation
# parch, family size highly correlated as expected

colormap = plt.cm.RdBu
plt.figure(figsize=(14,12))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(train.astype(float).corr(),linewidths=0.1,vmax=1.0, 
            square=True, cmap=colormap, linecolor='white', annot=True)

