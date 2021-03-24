'''
Utility functions
'''
import logging
from pathlib import Path

col_order = ['wid', 'fid', 'f1i', 'f2i', 'comp', 'y1', 'y2', 'year', 'year_1', 'year_2', 'year_start', 'year_end', 'year_start_1', 'year_end_1', 'year_start_2', 'year_end_2', 'weight', 'w1', 'w2', 'j', 'j1', 'j2', 'm', 'cs'].index

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

def logger_init(obj):
    '''
    Initialize logger.

    Arguments:
        obj (object): object requiring logger
    '''
    obj_name = type(obj).__name__.lower()
    # Begin logging
    obj.logger = logging.getLogger(obj_name)
    obj.logger.setLevel(logging.DEBUG)
    # Create logs folder
    Path('{}_logs'.format(obj_name)).mkdir(parents=True, exist_ok=True)
    # Create file handler which logs even debug messages
    fh = logging.FileHandler('{}_logs/{}_spam.log'.format(obj_name, obj_name))
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
