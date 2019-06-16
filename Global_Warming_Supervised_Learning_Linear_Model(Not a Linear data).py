#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 19:29:01 2019

@author: Suganth Mohan

Description : Used to Predit the weather of India with a simple linear regression model and 
see if it is poses a threat to humanity or atleast try to see. :-)


Modules Used:
    
    import json
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    from datetime import datetime
    
    
"""

# >>>>>>> ADD YOUR MODULES HERE
#### USED TO HANDLE THE JSON TYPE DATA
import json

#### USED FOR THE NUMPY ARRAY OPERATIONS
import numpy as np

#### USED FOR PLOTTING THE OBSERVATIONS INTO A GRAPH
import matplotlib.pyplot as plt

#### USED FOR DATAFRAMES CREATION AND HANDLING
import pandas as pd

#### USED FOR CREATING A TIME SERIES ON THE DATES
from datetime import datetime



def GetDataSource(*Retain_items):
    
    '''
    
    USED TO EXTRACT THE ITEMS FROM JSON FILE AND CREATE A DATAFRAME WITH THE DATASET
    
    ARGUMENTS :
        
        Retain_items -> used to retain the column names you set to retain before returning,
        
            constraint -> The headers must be an exact match, Otherwise brace yourselves for 
            errors
        
    '''    
    
    
    Source_Directory    = "/Users/Deepak/suganths_terminal/Suganth's Git/GlobalWarming"
    Source_File         = "Json_Monthly_Seasonal_Annual.json"
    Retain_items        = list(Retain_items)
    
    # CREATE A FILE HANDLE
    with open( Source_Directory + Source_File ) as readFileHandle:
        
        # READ THE RAW DATA AND LOAD IT WITH JSON MODULE
        raw_data = json.load(readFileHandle)
        
        # AS PER OUR REQUIREMENT WE WILL ONLY TAKE FROM YEAR TO ALL THE MONTHS
        
        # EXTRACT THE FIELDS LABELS ALONE AND CONVERT IT INTO LIST
        Fields_list = pd.io.json.json_normalize(raw_data['fields'])['label'].tolist()

        # CREATE THE DATASTRUCTURE OF DATAFRAME

        Weatheria_dataStruct = []

        # CREATE THE SKELETON FOR DATAFRAME
        for row in raw_data['data']:
            
            # CREATE DICTIONARY BY COMBINING TWO LISTS
            converted_row = dict(zip(Fields_list,row))
            
            Weatheria_dataStruct.append(converted_row)
         
        
        dataset = pd.DataFrame(Weatheria_dataStruct)

        
        if len(Retain_items) != 0:
            
            dataset = dataset[Retain_items]
                              
            return dataset

        else :
            
            return dataset


def ConvertMonthType(Month_):
    '''
    
    CovertMonthType ->  Is used to Convert the Month name into number format for creating
                        the time series.
                        
        arguments  : Month_
                It is used to get the month name in `String' and return the corresponding number of the
                month
        
        constraint :
                Cannot handle any other format otherthan specified below
                
                
    
    '''
    # CREATE CUSTOM MONTH CONVERSION
    my_month_list = {
                     
                    'JAN' : 1,
                    'FEB' : 2,
                    'MAR' : 3,
                    'APR' : 4,
                    'MAY' : 5,
                    'JUN' : 6,
                    'JUL' : 7,
                    'AUG' : 8,
                    'SEP' : 9,
                    'OCT' : 10,
                    'NOV' : 11,
                    'DEC' : 12                     
                    }
    
    # RETURN THE NUMBERICAL OF CUSTOM MONTH CONVERSION
    return my_month_list[Month_]

            
def CreateLinearDataset(Dataset_):
    '''
    
    CreateLinearDataset ->  is used to convert the multi field columns of months and year into
                            a linear dataset of each month along with the date
                            
        arguments : 
            Dataset_ -> It is the pandas dataframe dataset which we are going to convert into
                        a linear dataset form.
                        
                constraint : The function is written to handle a pandas dataframe dataset with
                             the matching formats, cannot handle other formats.
        
        returns   :     Returns the mutli-dimesional matrix with features in the form of a
                        ` Date ' and ` Celcius ' format.
    
    '''
    # CREATE A NEW DATAFRAME DATASTRUCTURE
    
    LinearDataSet = []
    
    # ITERATE THROUGH EACH AND EVERY ROWSET
    for index,Feature in Dataset_.iterrows():
        
        # CONVERT THE FEATURE INTO DICTIONARY
        New_Feature = dict(Feature)
    
        
        # ITERATE THROUGH THE MONTH SINCE EACH MONTH IS A KEY
        for column in New_Feature.keys():

            # CREATE LINEAR FEATURE
            Linear_Feature  = {}            
            
            # IF YEAR COLUMN APPEARS, SKIP IT
            if column == 'YEAR':
                continue
            
            # ADD DATE COLUMN VALUE            
            Linear_Feature['Date']    = datetime( int( Feature['YEAR'] ), int( ConvertMonthType(Month_ = column) ), 15)
            
            # ADD TEMPERATURE VALUE            
            Linear_Feature['Celcius'] = New_Feature[column]
            
            # APPEND THE NEW KEYS LIST TO THE NEW LINEAR DATASET
            LinearDataSet.append(Linear_Feature)
    
            

    # SORT THE DATAFRAME ACCORDING TO THE DATE FIELD, SINCE DICTIONARY LOSES ORDER AND RESET INDEX
    return pd.DataFrame(LinearDataSet)[['Date','Celcius']].sort_values('Date').reset_index(drop = True)
    
    
        

def main():
    
    
    # >>>>>>> PREPROCESSING PART
    
    #### GET THE DATASOURCE
    Weatheria = GetDataSource('YEAR','JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC')

    #### CONVERT THE MULTI DIMENSIONAL TABLE INTO MONTHLY FACTOR ENTRIES FOR OBSERVATION PURPOSES
    LinearDataSet = CreateLinearDataset(Dataset_ = Weatheria)
    
    #### SPLIT THE INDEPENDENT AND DEPENDENT VARIABLES
    
    ##### INDEPENDENT VARS
    independent_vars = LinearDataSet.iloc[:,0:1].values

    ##### DEPENDENT VARS
    dependent_vars   = LinearDataSet.iloc[:,1:2].values
    

    #### CONVERT THE DATA HERE
    
    ##### CONVERT THE TIME SERIES INTO LABELED VARIABLES
    from sklearn.preprocessing import LabelEncoder
    
    independent_LE = LabelEncoder()

    independent_vars = independent_LE.fit_transform(independent_vars[:,0])
    
    
    ###### RESHAPE THE INDEPENDENT VAR OF 1D ARRAY TO MATRIX TYPE HERE
    independent_vars = independent_vars.reshape((independent_vars.size,1))
    
    #### SINCE WE ARE USING A SUPERVISED LEARNING MODEL, WE WILL BE 
    #### USING TRAINING DATA AND TESTING DATA SPLITS
    from sklearn.cross_validation import train_test_split
        
    

    ##### SET THE TRAINING SIZE TO 80% which will be 0.8 
    ##### SO THE TEST SIZE WILL BE 1 - 0.8 = 0.2 ( 20% ) TO USE
    
    independent_train,independent_test,dependent_train,dependent_test = train_test_split(
                                                                        independent_vars,
                                                                        dependent_vars,
                                                                        test_size = 0.2,
                                                                        random_state = 0)

    
        
    # >>>>>>>> USE YOUR MODEL HERE
    
    #### USE LINEAR MODEL HERE (FOR NOW TO TEST)
    
    from sklearn.linear_model import LinearRegression
    
    regressor = LinearRegression()
    
    regressor.fit(independent_train,dependent_train)

    # >>>>>>>> PREDICT YOUR FUTURE VALUES HERE
    # Predicting the Test set results
    dependent_pred = regressor.predict(independent_test)

    
    # >>>>>>>> PLOT THE GRAPH OF YOUR MODEL HERE
    
    ##### VISUALIZE THE TRAINING MODEL RESULTS HERE
    # Visualising the Training set results
    plt.scatter(independent_train, dependent_train, color = 'red')
    plt.plot(independent_train, regressor.predict(independent_train), color = 'blue')
    plt.title('Date vs Temperature (Training set)')
    plt.xlabel('DATE')
    plt.ylabel('TEMPERATURE')
    plt.show()


    
    
    
if __name__ == '__main__':
    main()