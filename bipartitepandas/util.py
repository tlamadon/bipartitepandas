'''
Utility functions
'''
import logging
from pathlib import Path
# from numpy_groupies.aggregate_numpy import aggregate
import numpy as np
import pandas as pd
from statsmodels.stats.weightstats import DescrStatsW
import warnings

col_order = ['i', 'j', 'j1', 'j2', 'y', 'y1', 'y2', 't', 't1', 't2', 't1', 't2', 't11', 't12', 't21', 't22', 'w', 'w1', 'w2', 'g', 'g1', 'g2', 'm', 'cs'].index

def update_dict(default_params, user_params):
    '''
    Replace entries in default_params with values in user_params. This function allows user_params to include only a subset of the required parameters in the dictionary.

    Arguments:
        default_params (dict): default parameter values
        user_params (dict): user selected parameter values

    Returns:
        params (dict): default_params updated with parameter values in user_params
    '''
    params = default_params.copy()

    params.update(user_params)

    return params

def to_list(data):
    '''
    Convert data into a list if it isn't already.

    Arguments:
        data (obj): data to check if it's a list

    Returns:
        (list): data as a list
    '''
    if not isinstance(data, (list, tuple)):
        return [data]
    return data

loggers = {}

def logger_init(obj):
    '''
    Initialize logger.

    Arguments:
        obj (object): object requiring logger
    '''
    global loggers
    obj_name = type(obj).__name__.lower()

    if loggers.get(obj_name):
        # Prevent duplicate loggers
        # Source: https://stackoverflow.com/questions/7173033/duplicate-log-output-when-using-python-logging-module/55877763
        obj.logger = loggers.get(obj_name)
    else:
        # Begin logging
        obj.logger = logging.getLogger(obj_name)
        obj.logger.setLevel(logging.DEBUG)
        # Create logs folder
        Path('logs/{}_logs'.format(obj_name)).mkdir(parents=True, exist_ok=True)
        # Create file handler which logs even debug messages
        fh = logging.FileHandler('logs/{}_logs/{}_spam.log'.format(obj_name, obj_name))
        fh.setLevel(logging.DEBUG)
        # Create console handler with a higher log level
        ch = logging.StreamHandler()
        ch.setLevel(logging.ERROR)
        # Create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        # Add the handlers to the logger
        obj.logger.addHandler(fh)
        obj.logger.addHandler(ch)
        loggers[obj_name] = obj.logger

def col_dict_optional_cols(default_col_dict, user_col_dict, data_cols, optional_cols=()):
    '''
    Update col_dict to account for whether certain optional columns are included.

    Arguments:
        default_col_dict (dict): default col_dict values
        user_col_dict (dict): user col_dict
        data_cols (list): columns from user data
        optional_cols (list of lists): optional columns to check if included in user data. If sub-list has multiple columns, all columns must be included in the data for them to be added to new_col_dict

    Returns:
        new_col_dict (dict): updated col_dict
    '''
    if user_col_dict is None: # If columns already correct
        new_col_dict = default_col_dict
    else:
        new_col_dict = update_dict(default_col_dict, user_col_dict)
    # Add columns in 
    for col_list in to_list(optional_cols):
        include = True
        for col in to_list(col_list):
            exists_assigned = new_col_dict[col] is not None
            exists_not_assigned = (col in data_cols) and (new_col_dict[col] is None) and (col not in new_col_dict.values()) # Last condition checks whether data has a different column with same name
            if not exists_assigned and not exists_not_assigned:
                include = False
        if include:
            for col in to_list(col_list):
                if new_col_dict[col] is None:
                    new_col_dict[col] = col
        else: # Reset column names to None if not all essential columns included
            for col in to_list(col_list):
                new_col_dict[col] = None
    return new_col_dict

def aggregate_transform(frame, col_groupby, col_grouped, func, weights=None, col_name=None, merge=True):
    '''
    Adds transform to the numpy_groupies function aggregate. Source: https://stackoverflow.com/a/65967344.

    Arguments:
        frame (Pandas DataFrame): frame to transform into
        col_groupby (str): column to group by
        col_grouped (str): column to group
        func (function): function to apply. Can input a function, or a string for a pre-built function. Pre-built functions include:

            n_unique: number of unique values in col_grouped for each value in col_groupby

            max: max value

            min: min value

            sum: sum

            var: variance
        weights (str or None): weight column. Note that weights only work for 'sum' and 'var' options
        col_name (str or None): specify what to name the new column. If None and func is a string, sets col_name=func. If None and func is a function, sets col_name=m. If col_name is in frame's columns, adds 'm' until it will not overwrite an existing column
        merge (bool): if True, merge into dataframe

    Returns:
        if merge:
            (Pandas DataFrame): aggregated, transformed data merged into original dataframe
        else:
            (Pandas Series): aggregated, transformed data
    '''
    col1 = frame[col_groupby].to_numpy()
    col2 = frame[col_grouped].to_numpy()
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
        agg_array = np.split(col2, np.unique(col1, return_index=True)[1])[1:] # aggregate(col1, col2, 'array', fill_value=[])
    if isinstance(func, str):
        if col_name is None:
            col_name = func
        if col_name in frame.columns:
            while col_name in frame.columns:
                col_name += 'm'
        if func == 'array':
            pass
        elif func == 'n_unique':
            agg_array = np.array([len(np.unique(np.array(vals))) for vals in agg_array])
        elif func == 'max':
            agg_array = np.array([np.max(np.array(vals)) for vals in agg_array])
        elif func == 'min':
            agg_array = np.array([np.min(np.array(vals)) for vals in agg_array])
        elif func in ['sum', 'var']:
            if weights is not None:
                col_weights = frame[weights].to_numpy()
                agg_weights = np.split(col_weights, np.unique(col1, return_index=True)[1])[1:]
            else:
                agg_weights = np.ones(shape=len(agg_array))
            if func == 'sum':
                agg_array = np.array([np.sum(np.array(vals) * np.array(agg_weights[i])) for i, vals in enumerate(agg_array)])
            elif func == 'var':
                agg_array = np.array([DescrStatsW(np.array(vals), weights=np.array(agg_weights[i])).var for i, vals in enumerate(agg_array)])
        else:
            warnings.warn('Invalid function name, returning groupby with no function applied')
    else:
        if col_name in frame.columns:
            while col_name in frame.columns:
                col_name += 'm'
        agg_array = func(agg_array)

    if merge:
        agg_df = pd.DataFrame({col_groupby: np.unique(col1), col_name: agg_array}, index=np.arange(len(agg_array)))
        return frame[[col_groupby, col_grouped]].merge(agg_df, how='left', on=col_groupby)[col_name].to_numpy()
    return agg_array
