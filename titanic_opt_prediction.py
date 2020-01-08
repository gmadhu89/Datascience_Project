'''

Titanic: Predict the survivors

Optimization and EDA

https://www.kaggle.com/c/titanic/rules


'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import time

import os
import warnings
warnings.filterwarnings('ignore')


import seaborn as sns
sns.set(style="darkgrid")


from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import StandardScaler , MinMaxScaler , RobustScaler
from sklearn.model_selection import train_test_split , cross_val_score
from sklearn.metrics import confusion_matrix , roc_curve , auc
from sklearn import metrics




def elapsed(sec):

      '''

      This function returns the elapsed time

      '''

      if sec < 60:
            return str(round(sec)) +  ' secs'

      elif sec < 3600:
            return str(round((sec) / 60)) + ' mins'

      else:
            return str(round(sec / 3600)) + ' hrs'



def dispMissingVals(df):

    
    '''

      This function displays the missing values (NaN or Nulls)
      in the data sets

    '''

    for col in df.columns.tolist():
        print('{} column missing values: {}'.format(col , df[col].isnull().sum()))


    print('=============================')
    print()
    

def dataAnalysis(train , test):

    '''

    This function does data analysis and plotting
    of the dataset to understand the features better

    '''

    target = train['Survived']
    #sns.countplot(x = 'Sex' , hue = 'Survived' , data = train)

    '''

    fig , ax = plt.subplots(1 , 3 , figsize = (10 , 6))
    a = sns.countplot(x = 'Sex' , data = train , ax = ax[0] , order = ['male' , 'female'])
    b = sns.countplot(x = 'Sex' , data = train[target == 1] , ax = ax[1] , order = ['male' , 'female'])
    c = sns.countplot(x = 'Sex' , data= train[ ((train['Age'] < 21) & (train['Survived'] == 1)) ] , order = [1 , 2 , 3])

    ax[0].set_title('All passengers')
    ax[1].set_title('Survived Passenger')
    ax[2].set_title('Survived under age 21')

    '''
    
    #plt.show()



def featureEngineering(train , test):
      


    '''

    This function perfroms selection of relevant features
    handles missing or null values


    '''
    print('Missing values in training set')
    dispMissingVals(train)

    print('Missing values in test set')
    dispMissingVals(test)

    print()

    # Create new column deck from Cabin
    train['Deck'] = train['Cabin'].str.get(0)
    test['Deck'] = test['Cabin'].str.get(0)

    # Fill the missing values as NA
    train['Deck'] = train['Deck'].fillna('NA')
    test['Deck'] = test['Deck'].fillna('NA')

    '''

    fig , ax = plt.subplots(1 , 2 , figsize = (8 , 5))

    ta = sns.countplot(x = 'Deck' , data = train , ax = ax[0])
    tb = sns.countplot(x = 'Deck' , data = test , ax = ax[1])

    ax[0].set_title('Train Deck')
    ax[1].set_title('test Deck')

    plt.show()

    '''

    # Replace T with G
    train['Deck'].replace('T' , 'G' , inplace = True)

    # Drop the cabin column
    train.drop('Cabin' , axis = 1 , inplace = True)
    test.drop('Cabin' , axis = 1 , inplace = True)

    # Fill the embarked missing values with S - the most common occuring value
    train.loc[train['Embarked'].isna() , 'Embarked'] = 'S'

    # Fill the age missing values
    age_fill = train.groupby(['Pclass' , 'Sex' , 'Embarked'])[['Age']].median()

    #print(age_fill)
    #print()

    for cl in range(1 , 4):
        for sex in ['male' , 'female']:
            for em in ['C' , 'Q' , 'S']:

                val = pd.to_numeric(age_fill.xs(cl).xs(sex).xs(em).Age)

                train.loc[(train['Age'].isna() & (train['Pclass'] == cl)
                          & (train['Sex'] == sex) & (train['Embarked'] == em)), 'Age'] = val



                test.loc[(test['Age'].isna() & (test['Pclass'] == cl)
                          & (test['Sex'] == sex) & (test['Embarked'] == em)), 'Age'] = val                


    #print()
    #print(train.groupby(['Pclass' , 'Sex' , 'Embarked'])[['Age']].median())

    # Take numeric values for ticket
    train['Ticket'] = pd.to_numeric(train['Ticket'].str.split().str[-1] , errors = 'coerce')
    test['Ticket'] = pd.to_numeric(test['Ticket'].str.split().str[-1] , errors = 'coerce')

    # Fill missing values with median value
    train['Ticket'].fillna(train['Ticket'].median() , inplace = True)
    test['Fare'].fillna(train['Fare'].median() , inplace = True)

    train['Status'] = train['Name'].str.split(',').str.get(1).str.split('.').str.get(0).str.strip()
    test['Status'] = test['Name'].str.split(',').str.get(1).str.split('.').str.get(0).str.strip()

    # sns.countplot(x = 'Status' , data = train[(train['Survived'] == 1)])
    # plt.show()

    titles = ['Dr' , 'Rev' , 'Col' ,
                       'Major' , 'Mlle' , 'Don' ,
                       'Sir' , 'Ms' , 'Capt' ,
                       'Lady' , 'Mme' , 'the Countess' ,
                       'Jonkheer' , 'Dona']

    for t in titles:
          train['Status'].replace(t , 'IMP' , inplace = True)
          test['Status'].replace(t , 'IMP' , inplace = True)


    target = train['Survived']

    # Drop columns
    test.drop(['Name' , 'Ticket'] , axis = 1 , inplace = True)
    train.drop(['Survived' , 'Ticket' , 'Name'] , axis = 1 , inplace = True)

    cat_cols = ['Pclass' , 'Sex' , 'Embarked' , 'Status' , 'Deck']

    # Convert categorical cols to numeric
    train['Pclass'].replace({1 : 'A' , 2 : 'B' , 3 : 'C'} , inplace = True)
    test['Pclass'].replace({1 : 'A' , 2 : 'B' , 3 : 'C'} , inplace = True)

    #dispMissingVals(train)
    #dispMissingVals(test)

    train = pd.get_dummies(train , columns = cat_cols)
    test = pd.get_dummies(test , columns = cat_cols)

    #print(train.shape , test.shape)

    # Scale the data to remove bias from large values
    scaler = MinMaxScaler()
    train = scaler.fit_transform(train)
    test = scaler.transform(test)

    #print(train.head())
    



    return train , test , target
    

def applyML(train , test , target , idx):

      '''
       This function applies the ML algorithms
       and predicts for the test data

      '''
      # Random Forest Classifier
      rand_for_clf = RandomForestClassifier(bootstrap = True ,
                                            min_samples_leaf = 3 ,
                                            n_estimators = 5 ,
                                            min_samples_split = 10 ,
                                            max_features = 'sqrt' ,
                                            max_depth = 6)
      # CV Score for Random Forest
      rf_cv_score = cross_val_score(rand_for_clf , train , target , cv = 6)
      print('Random Forest: {}'.format(rf_cv_score))

      # Logistic Regression
      log_reg_clf = LogisticRegression()
      log_reg_cv_score = cross_val_score(log_reg_clf , train , target , cv = 6)
      print('Logistic Regression: {}'.format(log_reg_cv_score))

      # SVM
      svm_clf = SVC(C = 4)
      svm_cv_score = cross_val_score(svm_clf , train , target , cv = 5)
      print('SVM: {}'.format(svm_cv_score))

      # Predict using SVM
      svm_clf.fit(train , target)
      svm_pred = svm_clf.predict(test)

      # Create the result DataFrame
      result = pd.DataFrame({'PassengerId' : idx , 'Survived' : svm_pred})

      # Write the result to csv file
      result.to_csv('result_opt.csv' , index = False)

      print(result.head())
      
    

if __name__ == '__main__':
             

    start_time = time.time()


    # Load the dataset
    train_df = pd.read_csv('train.csv' , index_col = 'PassengerId')
    test_df = pd.read_csv('test.csv' , index_col = 'PassengerId')

    index = test_df.index
    
    # Analyse data
    dataAnalysis(train_df , test_df)

    # Handle Missing vals and drop cols
    train , test , target = featureEngineering(train_df , test_df)

    # Apply machine learning algorithms 
    applyML(train , test , target , index)
    
    elapsed_time = elapsed(time.time() - start_time)
    print('Elapsed time: ' , elapsed_time)  
