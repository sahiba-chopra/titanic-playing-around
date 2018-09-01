
# coding: utf-8

# In[40]:


from data_loader import *
from feature_pipeline import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import FeatureUnion
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn import linear_model
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostRegressor
import lightgbm as lgb
import xgboost as xgb


# In[2]:


class NumberSelector(BaseEstimator, TransformerMixin):
    """
    Transformer to select a single column from the data frame to perform additional transformations on
    Use on numeric columns in the data
    """
    def __init__(self, key):
        self.key = key

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[[self.key]]


# In[3]:


X_train.head()


# In[6]:


Pclass =  Pipeline([
                ('selector', NumberSelector(key='Pclass')),
                ('standard', StandardScaler())
            ])

ParchCat = Pipeline([
                ('selector', NumberSelector(key='ParchCat')),
                ('standard', StandardScaler())
            ])

FamilySize = Pipeline([
                ('selector', NumberSelector(key='FamilySize')),
                ('standard', StandardScaler())
            ])

IsAlone = Pipeline([
                ('selector', NumberSelector(key='IsAlone')),
                ('standard', StandardScaler())
            ])

Sex_Code = Pipeline([
                ('selector', NumberSelector(key='Sex_Code')),
                ('standard', StandardScaler())
            ])

Title_Code = Pipeline([
                ('selector', NumberSelector(key='Title_Code')),
                ('standard', StandardScaler())
            ])

Embarked_Code = Pipeline([
                ('selector', NumberSelector(key='Embarked_Code')),
                ('standard', StandardScaler())
            ])

AgeCat = Pipeline([
                ('selector', NumberSelector(key='AgeCat')),
                ('standard', StandardScaler())
            ])

FareCat = Pipeline([
                ('selector', NumberSelector(key='FareCat')),
                ('standard', StandardScaler())
            ])

Has_Cabin = Pipeline([
                ('selector', NumberSelector(key='Has_Cabin')),
                ('standard', StandardScaler())
            ])


# In[7]:


feats = FeatureUnion([('Pclass', Pclass), 
                      ('ParchCat', ParchCat),
                      ('FamilySize', FamilySize),
                      ('Has_Cabin', Has_Cabin),
                      ('Embarked_Code', Embarked_Code),
                      ('Title_Code', Title_Code),
                      ('IsAlone', IsAlone),
                      ('Sex_Code', Sex_Code),
                      ('AgeCat', AgeCat),
                      ('FareCat', FareCat)])

feature_processing = Pipeline([('feats', feats)])
feature_processing.fit_transform(X_train)


# In[31]:


# Random Forest

pipeline = Pipeline([
    ('features',feats),
    ('classifier', RandomForestClassifier(random_state = 42)),
])

pipeline.fit(X_train, Y_train)

preds = pipeline.predict(X_validation)
np.mean(preds == Y_validation)


# In[17]:


# list of hyperparameters to choose from
pipeline.get_params().keys()


# In[15]:


# Random Forest with hyperparameter tuning

hyperparameters = { 'classifier__max_depth': [50, 70],
                    'classifier__min_samples_leaf': [1,2]
                  }
clf = GridSearchCV(pipeline, hyperparameters, cv=5)
 
# Fit and tune model
clf.fit(X_train, Y_train)
clf.best_params_


# In[16]:


# 1% improvement
clf.refit

preds = clf.predict(X_validation)
probs = clf.predict_proba(X_validation)

np.mean(preds == Y_validation)


# In[23]:


# ExtraTrees 

pipeline = Pipeline([
    ('features',feats),
    ('classifier', ExtraTreesClassifier(random_state = 42)),
])

pipeline.fit(X_train, Y_train)

preds = pipeline.predict(X_validation)
np.mean(preds == Y_validation)


# In[25]:


# Decision Trees - better than random forest

pipeline = Pipeline([
    ('features',feats),
    ('classifier', DecisionTreeClassifier(random_state = 42)),
])

pipeline.fit(X_train, Y_train)

preds = pipeline.predict(X_validation)
np.mean(preds == Y_validation)


# In[28]:


# Gradient Boosting

pipeline = Pipeline([
    ('features',feats),
    ('classifier', GradientBoostingClassifier(random_state = 42)),
])

pipeline.fit(X_train, Y_train)

preds = pipeline.predict(X_validation)
np.mean(preds == Y_validation)


# In[30]:


# Logistic Regression - does best

pipeline = Pipeline([
    ('features',feats),
    ('classifier', LogisticRegression(random_state = 42)),
])

pipeline.fit(X_train, Y_train)

preds = pipeline.predict(X_validation)
np.mean(preds == Y_validation)


# In[35]:


# Catboost

pipeline = Pipeline([
    ('features',feats),
    ('classifier', CatBoostRegressor(random_state = 42)),
])

pipeline.fit(X_train, Y_train)

preds = pipeline.predict(X_validation)
np.mean(preds == Y_validation)


# In[39]:


# light gbm

pipeline = Pipeline([
    ('features',feats),
    ('classifier', lgb.LGBMClassifier(random_state = 42)),
])

pipeline.fit(X_train, Y_train)

preds = pipeline.predict(X_validation)
np.mean(preds == Y_validation)


# In[41]:


# xgboost

pipeline = Pipeline([
    ('features',feats),
    ('classifier', xgb.XGBClassifier(random_state = 42)),
])

pipeline.fit(X_train, Y_train)

preds = pipeline.predict(X_validation)
np.mean(preds == Y_validation)

