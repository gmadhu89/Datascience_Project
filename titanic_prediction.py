'''

Titanic: Machine Learning from Disaster

https://www.kaggle.com/c/titanic/rules


'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
sns.set(style="darkgrid")


from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import metrics



def concatDataSets(train , test):

      '''

       This function combines the train and test
       datasets

      '''

      total_df = pd.concat([train , test] , sort = True).reset_index(drop = True)
      return total_df



def splitDataSets(total , train_l):

      '''

      This function splits the dataset into train and test
      In the test dataset, the class variable is dropped

      '''

      return total.loc[ : train_l - 1] , total.loc[train_l : ].drop(['Survived'] , axis = 1)


def dispMissing(df):

      '''

      This function displays the missing values (NaN or Nulls)
      in the data sets

      '''

      for col in df.columns.tolist():
            print('{} column missing values: {}'.format(col , df[col].isnull().sum()))

      print()


def getFeatures(df):

      '''

      This function does the necessary pre-processing
      and splits the features from class labels

      '''

      features = df.drop(columns = ['Name' , 'Survived' , 'Ticket'] , axis = 1)
      labels = df['Survived']

      return features , labels


def evaluateModel(model , model_nm , x , y , test):

      '''

      This function evaluates the dataset on various
      classification algorithms and computes performance
      metrics

      '''
      # Create train and test splits
      x_train , x_test , y_train , y_test = train_test_split(x.values,
                                                             y ,
                                                             test_size = 0.40 ,
                                                             random_state = 30)
      # Fit the train features and labels
      model.fit(x_train , y_train)

      # Prdict on the test data
      y_pred = model.predict(x_test)

      #print(y_pred)

      # Get the accuracy
      acc = np.mean(y_pred == y_test)
      sk_acc = metrics.accuracy_score(y_test, y_pred)
      confusion = metrics.confusion_matrix(y_test , y_pred)
      TP = confusion[1 , 1]
      TN = confusion[0 , 0]
      FP = confusion[0 , 1]
      FN = confusion[1 , 0]
      conf_acc = (TP + TN) / float(TP + TN + FP + FN)
      
      print('{} Accuracy: {:5.2f}%'.format(model_nm , acc * 100))
      print('{} SKLearn Acc: {:5.2f}%'.format(model_nm , sk_acc * 100))
      print('{} Confusion Matrix Acc: {:5.2f}%'.format(model_nm , conf_acc * 100))
      print('================================================')
      print()


      # predict the test file
      if model_nm == 'Random Forest':
            survived_pred = model.predict(test)
            survived_pred = list(map(int , survived_pred.tolist()))

            survived_df = pd.DataFrame(survived_pred , columns = ['Survived'])

            #print(len(survived_pred) , type(survived_pred))
            #print(survived_df)

            #print(test['PassengerId'])

            test['Survived'] = survived_df.values

            #result_df = pd.concat([test['PassengerId'] , survived_df] , ignore_index = True , axis = 1)

            print(test.columns.tolist())

            print(test[['PassengerId' , 'Survived']])

            # Save to csv
            test[['PassengerId' , 'Survived']].to_csv('titanic_result.csv', index = False)

      return y_pred

      

def SKModels(total_df):

      '''

       This function applies the Gaussian Naive Bayes
       on the data set

      '''

      df_train , df_test = splitDataSets(total_df , train_l)


      # Drop 'Name' from test
      df_test = df_test.drop(columns = ['Name' , 'Ticket'] , axis = 1)

      print('Test Columns: ' , df_test.columns.tolist())

      # Seperate the Features from the target class variable
      x_df, y_labels = getFeatures(df_train)

      #print(x_df.columns.tolist())
      #print(y_labels)

      # Create the GaussNB Object
      gauss_NB = GaussianNB()

      # Create Logistic Regression Object
      log_reg = LogisticRegression(solver = 'lbfgs' , max_iter = 600)

      # Create Decision Tree classifier
      dec_tree = DecisionTreeClassifier()

      # Create a Random Forest Classifier
      rand_forest = RandomForestClassifier(n_estimators  = 100)

      # Evaluate the model
      nb_pred = evaluateModel(gauss_NB , 'Naive Bayes' , x_df , y_labels , df_test)
      log_reg_pred = evaluateModel(log_reg , 'Logistic Regression' , x_df , y_labels, df_test)
      dec_tree_pred = evaluateModel(dec_tree , 'Decision Tree' , x_df , y_labels , df_test)
      rand_for_pred = evaluateModel(rand_forest , 'Random Forest' , x_df , y_labels , df_test)

      # Predict for the test file
      
      


if __name__ == '__main__':

      # Load the dataset
      train_df = pd.read_csv('train.csv')
      test_df = pd.read_csv('test.csv')

      train_l , test_l = len(train_df) , len(test_df)

      print('Training records {} ,  Test records {}'.format(train_l , test_l))

      total_df = concatDataSets(train_df , test_df)

      print()
      print('Training Set: ')
      dispMissing(train_df)

      print('Test Set: ')
      dispMissing(test_df)

      # Show correlation
      df_corr_all = total_df.corr().abs().unstack().sort_values(
            kind = 'quicksort' , ascending = False).reset_index()


      df_corr_all.rename(columns = {'level_0' : 'Feature 1' , 'level_1' : 'Feature 2' , 0 : 'Corr COeff'}, inplace = True)
      

      print(df_corr_all[df_corr_all['Feature 1'] == 'Age'])

      # Fill missing values by grouping 'Sex' and 'PClass'
      # and using median of age
      total_df['Age'] = total_df.groupby(['Sex' , 'Pclass'])['Age'].apply(lambda x: x.fillna(x.median()))

      # Convert categorical to numeric
      # Sex is male (0) and female (1)
      
      total_df['Sex'] = np.where(total_df['Sex'] == 'male' , 0 , 1)
      print()
      print(total_df[total_df['Embarked'].isnull()])

      # Fill NA values for other columns
      total_df['Embarked'] = total_df['Embarked'].fillna('S')
      print(total_df[total_df['Fare'].isnull()])
      # Embarked S = 0 , C = 1 , Q = 2
      total_df['Embarked'] = np.where(total_df['Embarked'] == 'S' , 1,
                                      np.where(total_df['Embarked'] == 'C' , 2,
                                               np.where(total_df['Embarked'] == 'Q' , 3 , 4)
                                               ))

      # Fill the median fare value of Pclass = 3
      # Parh = 0 and Sibsp = 0
      med_fare = total_df.groupby(['Pclass' , 'Parch' , 'SibSp'])['Fare'].median()[3][0][0]
      print(med_fare)
      total_df['Fare'] = total_df['Fare'].fillna(med_fare)


      # Drop the cabin feature
      total_df.drop(['Cabin'] , inplace = True , axis = 1)
      dispMissing(total_df)

      '''

      df_train , df_test = splitDataSets(total_df , train_l)

      dispMissing(df_train)
      print()
      dispMissing(df_test)
      print()
      
      '''
      # Apply ML Models
      #print('Gaussian Naive Bayes.....')
      SKModels(total_df)

      
