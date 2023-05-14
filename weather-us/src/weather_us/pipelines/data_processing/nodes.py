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
import pandas as pd
from sklearn import preprocessing


def extract_training_data(df):
    """" Extracting the training data from raw data and droping Date column
    Args: Pandas DataFrame here Raw Data
    Return: DataFrame i.e. Training Data"""
    df = df.drop("Date", axis = 1)
    df1 = df[df['RainTomorrow'].notna()]   #This is our training data
    return df1

def label_encoding_filling_null(df1):
    """" label encoding the training  and filling null values in column
    Args:  DataFrame training data
    Return: DataFrame without null values i.e. Training Data"""
    #replacing null values in categorical variables with mode value
    df1['WindGustDir'].fillna(df1['WindGustDir'].mode()[0], inplace=True)
    df1['WindDir9am'].fillna(df1['WindDir9am'].mode()[0], inplace=True)
    df1['WindDir3pm'].fillna(df1['WindDir3pm'].mode()[0], inplace=True)
    df1['RainToday'].fillna(df1['RainToday'].mode()[0], inplace=True)
    df1['RainTomorrow'].fillna(df1['RainTomorrow'].mode()[0], inplace=True)

        # label_encoder object knows how to understand word labels.
    label_encoder = preprocessing.LabelEncoder()
    
    # Encode labels in column 'Location'.'RainTomorrow','Date','WindDir9am',	'WindDir3pm',	'RainToday',	'RainTomorrow'
    df1['Location']= label_encoder.fit_transform(df1['Location'])
    df1['WindGustDir']= label_encoder.fit_transform(df1['WindGustDir'])
    df1['WindDir9am']= label_encoder.fit_transform(df1['WindDir9am'])
    df1['WindDir3pm']= label_encoder.fit_transform(df1['WindDir3pm'])
    df1['RainToday']= label_encoder.fit_transform(df1['RainToday'])
    df1['RainTomorrow']= label_encoder.fit_transform(df1['RainTomorrow'])
    
    #filling the null values in numerical variable with median
    df2 = df1.fillna(df1.median())


    return df2