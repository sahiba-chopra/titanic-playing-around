
# coding: utf-8

# In[1]:


from data_loader import *
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# In[2]:


full_data = [train,test]

# Define function to extract titles from passenger names
def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""
# Create a new feature Title, containing the titles of passenger names
for dataset in full_data:
    dataset['Title'] = dataset['Name'].apply(get_title)
# Group all non-common titles into one single grouping "Rare"
for dataset in full_data:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

# Create new feature FamilySize as a combination of SibSp and Parch
for dataset in full_data:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
# Create new feature IsAlone from FamilySize
for dataset in full_data:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1




train.head()





lb_make = LabelEncoder()




for dataset in full_data:
    # encode sex
    dataset['Sex_Code'] = lb_make.fit_transform(dataset["Sex"])
    # encode title
    dataset['Title_Code'] = lb_make.fit_transform(dataset["Title"])
    # encode port of embarkment
    dataset['Embarked_Code'] = lb_make.fit_transform(dataset["Embarked"].astype(str))
    # create 4 age buckets
    dataset['AgeCat'] = pd.cut(dataset['Age'], 4)
    dataset['AgeCat'] = lb_make.fit_transform(dataset["AgeCat"].astype(str))
    # create 4 fare buckets
    dataset['FareCat'] = pd.cut(dataset['Fare'], 4)
    dataset['FareCat'] = lb_make.fit_transform(dataset["FareCat"].astype(str))    
    dataset['Has_Cabin'] = dataset["Cabin"].apply(lambda x: 0 if type(x) == float else 1)
    # create 3 parch buckets
    dataset['ParchCat'] = pd.cut(dataset['Parch'], 3)
    dataset['ParchCat'] = lb_make.fit_transform(dataset["ParchCat"].astype(str))    





PassengerID = test['PassengerId']


# In[102]:


drop_elements = ['PassengerId','Name', 'Sex','Ticket','Fare','Embarked','Cabin','Age','SibSp','Title','Parch']
train = train.drop(drop_elements, axis = 1)
test  = test.drop(drop_elements, axis = 1)





# columns are integers
train.dtypes


# creating validation set

features= [c for c in train.columns.values if c  not in ['Survived']]
target = 'Survived'

X_train, X_validation, Y_train, Y_validation = train_test_split(train[features], train[target], test_size=0.33, random_state=42)
X_train.head()


# since we're using kaggle data there's no 'Survived' column
X_test = test[features]


# In[3]:


train.head()

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
