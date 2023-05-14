# Copyright 2021 QuantumBlack Visual Analytics Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND
# NONINFRINGEMENT. IN NO EVENT WILL THE LICENSOR OR OTHER CONTRIBUTORS
# BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF, OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# The QuantumBlack Visual Analytics Limited ("QuantumBlack") name and logo
# (either separately or in combination, "QuantumBlack Trademarks") are
# trademarks of QuantumBlack. The License does not grant you any right or
# license to the QuantumBlack Trademarks. You may not use the QuantumBlack
# Trademarks or any confusingly similar mark as a trademark for your product,
#     or use the QuantumBlack Trademarks in any other manner that might cause
# confusion in the marketplace, including but not limited to in advertising,
# on websites, or on software.
#
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
from typing import Dict, Tuple

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import preprocessing
import matplotlib.pyplot as plt # data visualization
import seaborn as sns # statistical data visualization
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
#import classification regression
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')


# We need to create a node
def  split_data(df2):

    """Splits data into features and targets training and test sets.
    Args:
        data: Data containing features and target.
    Returns:
        Split data.
    """
    X = df2.drop(['RainTomorrow'], axis=1)

    y = df2['RainTomorrow']
    
    # split X and y into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
    
    return X_train, X_test, y_train, y_test


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> LogisticRegression:
    """Trains the logistic regression model.
    Args:
        X_train: Training data of independent features.
        y_train: Training data for price.
    Returns:
        Trained model.
    """
    # instantiate the model
    logreg = LogisticRegression(solver='liblinear', random_state=0)
    # fit the model
    logreg.fit(X_train, y_train)
    return logreg


def evaluate_model(
    logreg: LogisticRegression, X_test: pd.DataFrame, y_test: pd.Series
):
    """Calculates and logs the coefficient of determination.
    Args:
        logreg: Trained model.
        X_test: Testing data of independent features.
        y_test: Testing data for price.
        
    Returns:
        y_pred_test:prediction on x test
        acc        :accuracy_score
    """
    y_pred_test = logreg.predict(X_test)
    #**Check accuracy score**
    acc=(accuracy_score(y_test, y_pred_test))
    print('Model accuracy score: ', acc)
    return y_pred_test,acc



def train_model_rf(X_train: pd.DataFrame, y_train: pd.Series) :
    """Trains the RandomForestClassifier.
    Args:
        X_train: Training data of independent features.
        y_train: Training data for price.
    Returns:
        Trained model.
    """
    # instantiate the model
    random_forest = RandomForestClassifier(n_estimators=100)

   # fit the model
    random_forest.fit(X_train, y_train)
    return random_forest

def evaluate_model_rf(
     random_forest :RandomForestClassifier(),X_test: pd.DataFrame, y_test: pd.Series
):
    """Calculates and logs the coefficient of determination.
    Args:
        random_forest: Trained model.
        X_test: Testing data of independent features.
        y_test: Testing data for price.
        
    Returns:
        y_pred_test:prediction on x test
        acc        :accuracy_score
    """
    y_pred_test =  random_forest.predict(X_test)
    #**Check accuracy score**
    
    acc=(accuracy_score(y_test, y_pred_test))
    print('Model accuracy score: ', acc)
    return(y_pred_test,acc)



def train_model_dt(X_train: pd.DataFrame, y_train: pd.Series)  :
    """Trains the DecisionTreeClassifier model.
    Args:
        X_train: Training data of independent features.
        y_train: Training data for price.
    Returns:
        Trained model.
    """
    # instantiate the model
    decision_tree = DecisionTreeClassifier()

   # fit the model
    decision_tree.fit(X_train, y_train)
    return  decision_tree

def evaluate_model_dt(
     decision_tree : DecisionTreeClassifier(),X_test: pd.DataFrame, y_test: pd.Series
):
    """Calculates and logs the coefficient of determination.
    Args:
        decision_tree: Trained model.
        X_test: Testing data of independent features.
        y_test: Testing data for price.
        
    Returns:
        y_pred_test:prediction on x test
        acc        :accuracy_score
    """
    y_pred_test =   decision_tree.predict(X_test)
    #**Check accuracy score**
    acc=(accuracy_score(y_test, y_pred_test))
    print('Model accuracy score: ', acc)
    return(y_pred_test,acc)


def train_model_knn(X_train: pd.DataFrame, y_train: pd.Series)  :
    """Trains the KNeighborsClassifier Model.
    Args:
        X_train: Training data of independent features.
        y_train: Training data for price.
    Returns:
        Trained model.
    """
    # instantiate the model
    knn = KNeighborsClassifier(n_neighbors = 3)

   # fit the model
    knn.fit(X_train, y_train)
    return  knn

def evaluate_model_knn(
      knn : KNeighborsClassifier(),X_test: pd.DataFrame, y_test: pd.Series
):
    """Calculates and logs the coefficient of determination.
    Args:
        knn: Trained model.
        X_test: Testing data of independent features.
        y_test: Testing data for price.
        
    Returns:
        y_pred_test:prediction on x test
        acc        :accuracy_score
    """
    y_pred_test =   knn.predict(X_test)
    #**Check accuracy score**
    acc=(accuracy_score(y_test, y_pred_test))
    print('Model accuracy score: ', acc)
    return(y_pred_test,acc)

