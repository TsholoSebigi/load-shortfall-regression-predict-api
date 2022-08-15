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
    #label
    map_sps                         = {'sp1':1,'sp2':2,'sp3':3,'sp4':4,'sp5':5,'sp6':6,'sp7':7,'sp8':8,'sp9':9,'sp10':10,'sp11':11,
                                   'sp12':12,'sp13':13,'sp14':14,'sp15':15,'sp16':16,'sp17':17,'sp18':18,'sp19':19,'sp20':20  ,
                                   'sp21':21,'sp22':22,'sp23':23,'sp24':24,'sp25':25}
    map_levels                      = {'level_1':1,'level_2':2,'level_3':3,'level_4':4,'level_5':5,'level_6':6,'level_7':7,
                                       'level_8':8,'level_9':9,'level_10':10}
    map_weekdays                    = {"Mon": 0,"Tue": 1,"Wed": 2,"Thu": 3,"Fri": 4,"Sat": 5,"Sun": 6}

    trainLabel                      = feature_vector_df.copy()
    trainLabel['Time']              = pd.to_datetime(trainLabel['time'])
    trainLabel['Month']             = [i.month for i in trainLabel['Time']]
    trainLabel['quarter']           = [i.quarter for i in trainLabel['Time']]
    trainLabel['Year']              = [i.year  for i in trainLabel['Time']]
    trainLabel['Hour']              = [i.hour  for i in trainLabel['Time']]
    trainLabel['partOfDay']         = trainLabel['Hour'].apply(lambda x: partOfDay(x))
    trainLabel['dayOfWeek']         = trainLabel['Time'].apply(lambda x: x.strftime("%a"))
    trainLabel['Weekday']           = trainLabel["dayOfWeek"].map(map_weekdays)
    trainLabel['Seville_pressure']  = trainLabel["Seville_pressure"].map(map_sps)
    trainLabel['Valencia_wind_deg'] = trainLabel["Valencia_wind_deg"].map(map_levels)
    trainLabel['isWeekday']         = trainLabel["Weekday"].apply(lambda x: 1 if x < 5 else 0)
    trainLabel                      = trainLabel.drop(['time','Time','Unnamed: 0','dayOfWeek','Hour'],axis=1)



    ############################################################################
    trainLabel['Valencia_pressure'].fillna(1012.0514065222828, inplace=True)

    
    print(trainLabel)
    
    #normalize, crude at this point
    scalerOneHot      = StandardScaler()
    XtrainOneHotStd   = scalerOneHot.fit_transform(trainLabel)
    XtrainOneHotStdDf = pd.DataFrame(XtrainOneHotStd,columns=trainLabel.columns)

    print('made it here')
    
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
    return prediction.tolist()
