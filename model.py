"""

    Helper functions for the pretrained model to be used within our API.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.

    Importantly, you will need to modify this file by adding
    your own data preprocessing steps within the `_preprocess_data()`
    function.
    ----------------------------------------------------------------------

    Description: This file contains several functions used to abstract aspects
    of model interaction within the API. This includes loading a model from
    file, data preprocessing, and model prediction.  

"""

# Helper Dependencies

import pickle
import json
import datetime
import warnings
import time
import numpy                     as np
import pandas                    as pd
from   sklearn.preprocessing     import StandardScaler
from   sklearn                   import linear_model




def random_imputation(df, feature):

    number_missing = df[feature].isnull().sum()
    observed_values = df.loc[df[feature].notnull(), feature]
    df.loc[df[feature].isnull(), feature + '_imp'] = np.random.choice(observed_values, number_missing, replace = True)
    
    return df

#Using this labeling because peak hours in the day have higher usage than others, it is 3 hourly data
#so could have just put 0,3,6... Also weekdays and weekends tend to have different usages
def partOfDay(hour):
    hour = int(hour)
    if hour >= 0 and hour <= 3:
        return 0
    if hour > 3 and hour <= 6:
        return 1
    if hour > 6 and hour <= 9:
        return 2
    if hour > 9 and hour <= 12:
        return 3
    if hour >12 and hour <= 15:
        return 4
    if hour >15 and hour <= 18:
        return 5
    if hour >18 and hour <= 21:
        return 6
    if hour >21 and hour < 24:
        return 7

def _preprocess_data(data):
    """Private helper function to preprocess data for model prediction.

    NB: If you have utilised feature engineering/selection in order to create
    your final model you will need to define the code here.


    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.

    Returns
    -------
    Pandas DataFrame : <class 'pandas.core.frame.DataFrame'>
        The preprocessed data, ready to be used our model for prediction.
    """
    # Convert the json string to a python dictionary object
    feature_vector_dict = json.loads(data)
    # Load the dictionary as a Pandas DataFrame.
    feature_vector_df = pd.DataFrame.from_dict([feature_vector_dict])

    # ---------------------------------------------------------------
    # NOTE: You will need to swap the lines below for your own data
    # preprocessing methods.
    #
    # The code below is for demonstration purposes only. You will not
    # receive marks for submitting this code in an unchanged state.
    # ---------------------------------------------------------------

    # ----------- Replace this code with your own preprocessing steps --------
    #one hot encoding
    trainOneHot              = pd.get_dummies(feature_vector_df, columns=['Valencia_wind_deg','Seville_pressure'], drop_first=True)
    trainOneHot['Time']      =  pd.to_datetime(trainOneHot['time'])
    trainOneHot['Year']      = [i.year  for i in trainOneHot['Time']]
    trainOneHot['Month']     = trainOneHot['Time'].apply(lambda x: x.strftime("%b")) 
    trainOneHot['dayOfWeek'] = trainOneHot['Time'].apply(lambda x: x.strftime("%a"))
    trainOneHot['Hour']      = [i.hour  for i in trainOneHot['Time']]
    trainOneHot['partOfDay'] = trainOneHot['Hour'].apply(lambda x: partOfDay(x))
    trainOneHot['quarter']   = [i.quarter for i in trainOneHot['Time']]
    trainOneHot['isWeekday'] = trainOneHot['Time'].apply(lambda x: 0 if x in ['Sat','Sun'] else 1)
    trainOneHot              = pd.get_dummies(trainOneHot, columns=['dayOfWeek','Month','quarter','partOfDay'], drop_first=True)
    trainOneHot              = trainOneHot.drop(['time','Time','Unnamed: 0'],axis=1)


    ############################################################################
    #Impute
    missing_columns = ['Valencia_pressure'] # there could be multiple missing columns in other datasets
    
    for feature in missing_columns:
        trainOneHot[feature + '_imp'] = trainOneHot[feature]
        trainOneHot                   = random_imputation(trainOneHot, feature)
        
    
    random_data = pd.DataFrame(columns = ["Ran" + name for name in missing_columns])
    
    for feature in missing_columns:
           
        random_data["Ran" + feature] = trainOneHot[feature + '_imp']
    
        parameters     = list(set(trainOneHot.columns) - set(missing_columns) - {feature + '_imp'})
        model          = linear_model.LinearRegression()
        model.fit(X    = trainOneHot[parameters], y = trainOneHot[feature + '_imp'])
        
        #Standard Error of the regression estimates is equal to std() of the errors of each estimates
        predict        = model.predict(trainOneHot[parameters])
        std_error      = (predict[trainOneHot[feature].notnull()] - trainOneHot.loc[trainOneHot[feature].notnull(), feature + '_imp']).std()
        
        #preserve the index of the missing data from the original dataframe
        random_predict = np.random.normal(size  = trainOneHot[feature].shape[0], 
                                         loc   = predict                      , 
                                          scale = std_error                    )
       
        #random_data.loc[(trainOneHot[feature].isnull()) & (random_predict > 0), "Ran" + feature] = random_predict[(trainOneHot[feature].isnull()) & (random_predict > 0)]
        random_data.loc[(trainOneHot[feature].isnull()) & (random_predict > 0), "Ran" + feature] = trainOneHot['Valencia_pressure'].mean()
        
    #replace with imputed data and                                                                                                               
    trainOneHot['Valencia_pressure'] = random_data['RanValencia_pressure']

    print(trainOneHot.columns)
    
    normalize, crude at this point
    scalerOneHot      = StandardScaler()
    XtrainOneHotStd   = scalerOneHot.fit_transform(trainOneHot)
    XtrainOneHotStdDf = pd.DataFrame(XtrainOneHotStd,columns=trainOneHot.columns)

    
    #predict_vector = feature_vector_df[['Madrid_wind_speed','Bilbao_rain_1h','Valencia_wind_speed']]
    # ------------------------------------------------------------------------

    return XtrainOneHotStd

def load_model(path_to_model:str):
    """Adapter function to load our pretrained model into memory.

    Parameters
    ----------
    path_to_model : str
        The relative path to the model weights/schema to load.
        Note that unless another file format is used, this needs to be a
        .pkl file.

    Returns
    -------
    <class: sklearn.estimator>
        The pretrained model loaded into memory.

    """
    return pickle.load(open(path_to_model, 'rb'))


""" You may use this section (above the make_prediction function) of the python script to implement 
    any auxiliary functions required to process your model's artifacts.
"""

def make_prediction(data, model):
    """Prepare request data for model prediction.

    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.
    model : <class: sklearn.estimator>
        An sklearn model object.

    Returns
    -------
    list
        A 1-D python list containing the model prediction.

    """
    # Data preprocessing.
    prep_data = _preprocess_data(data)
    print(prep_data)
    print(len(prep_data))
    # Perform prediction with model and preprocessed data.
    prediction = model.predict(prep_data)
    # Format as list for output standardisation.
    return prediction[0].tolist()
