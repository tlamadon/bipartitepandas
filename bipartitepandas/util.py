'''
Utility functions.
'''
import logging
from pathlib import Path
from collections.abc import MutableMapping
# from numpy_groupies.aggregate_numpy import aggregate
import numpy as np
import pandas as pd
from statsmodels.stats.weightstats import DescrStatsW
import types
import warnings

def _text_num_split(col_name):
    '''
    Split column name into string and number components. Source: https://stackoverflow.com/a/55843049/17333120.

    Arguments:
        col_name (str): column name

    Returns:
        (list): first entry is column name string component; second entry is column number component
    '''
    for index, char in enumerate(col_name):
        if char.isdigit():
            return (col_name[: index], col_name[index:])
    return (col_name, '')

def _sort_cols(cols):
    '''
    Sort columns. Prioritize default columns, then sort alphabetically.

    Arguments:
        cols (list of str): list of column names to be sorted

    Returns:
        (list): sorted columns
    '''
    default_cols = ['i', 'j', 'y', 't', 'g', 'w', 'm'] #, 'cs', 'l', 'k', 'alpha', 'psi', 'alpha_hat', 'psi_hat']

    default_cols_dict = {}
    custom_cols_dict = {}

    for col in cols:
        col_str, col_num = _text_num_split(col)
        if col_str in default_cols:
            # If column is a default column
            if col_str in default_cols_dict.keys():
                # If column already seen
                default_cols_dict[col_str].append(col)
            else:
                # If column not already seen
                default_cols_dict[col_str] = [col]
        else:
            # If column is a custom column
            if col_str in custom_cols_dict.keys():
                # If column already seen
                custom_cols_dict[col_str].append(col)
            else:
                # If column not already seen
                custom_cols_dict[col_str] = [col]

    # Sort default and custom columns
    sorted_default_cols = [sorted_default_col for default_col in sorted(default_cols_dict.keys(), key=default_cols.index) for sorted_default_col in sorted(default_cols_dict[default_col])]
    sorted_custom_cols = [sorted_custom_col for custom_col in sorted(custom_cols_dict.keys()) for sorted_custom_col in sorted(custom_cols_dict[custom_col])]

    return sorted_default_cols + sorted_custom_cols

# Source: https://stackoverflow.com/a/54009756/17333120
fn_type = (types.BuiltinFunctionType, types.FunctionType, types.MethodType)

col_type = (np.ndarray, pd.Series)

def _is_subtype(obj, types):
    '''
    Check if obj is a subtype of types. Can also input 'fn' to specify the tuple of (types.BuiltinFunctionType, types.FunctionType, types.MethodType), as these types are not pickleable so multiprocessing fails if they are input directly.

    Arguments:
        obj (object): object to check
        types (type or string, or list of types or strings): types to check

    Returns:
        (bool): if subtype, returns True
    '''
    obj_type = type(obj)
    for type_i in to_list(types):
        if isinstance(type_i, str):
            if type_i == 'fn':
                if _is_subtype(obj, fn_type):
                    return True
            else:
                raise ValueError(f"Only valid string input is 'fn', but input included string {type_i!r}.")
        else:
            if np.issubdtype(obj_type, type_i):
                return True
    return False

def _is_subdtype(col, types):
    '''
    Check if column dtype is a subtype of types. Source: https://stackoverflow.com/a/40312924/17333120.

    Arguments:
        col (NumPy Array or Pandas Series): column to compare
        types (str or list of str): types to check (options are 'float', 'int', 'any', and 'categorical')

    Returns:
        (bool): if subtype, returns True
    '''
    # Get dtype kind of current column
    col_dtype_kind = col.dtype.kind
    # Get typecodes for dtype kinds (typecodes are 'Character', 'Integer', 'UnsignedInteger', 'Float', 'Complex', 'AllInteger', 'AllFloat', 'Datetime', and 'All)
    typecodes = np.typecodes
    # Add Int64 to typecodes
    typecodes['Integer'] += 'Int64'
    typecodes['AllInteger'] += 'Int64'
    typecodes['All'] += 'Int64'
    # Dictionary linking input types to typecodes
    type_str_to_typecodes_dict = {
        'float': ['AllFloat', 'AllInteger'],
        'int': 'AllInteger',
        'any': 'All',
        'categorical': 'All'
    }
    for type_i in to_list(types):
        # Check whether column is valid type
        for sub_type_str_to_typecode in to_list(type_str_to_typecodes_dict[type_i]):
            if col_dtype_kind in typecodes[sub_type_str_to_typecode]:
                return True
    return False

class ParamsDict(MutableMapping):
    '''
    Dictionary with fixed keys, and where values must follow given rules. Source: https://stackoverflow.com/a/14816620/17333120.

    Arguments:
        default_dict (dict): default dictionary. Each key should provide a tuple of (default_value, options_type, options, description, constraints), where `default_value` gives the default value associated with the key; `options_type` defines the valid types the value may take; `options` gives the valid values that can associated with the key (this can be a type or a set of particular values); `description` gives a description of the key-value pair; and `constraints` is either None if there are no constraints, or gives a description of the constraints on the value. The specifics on `options_type` follow. If `options_type` is:
            'type' - the key must be associated with a particular type
            'list_of_type' - the value must be a particular type or a list of values of a particular type
            'type_none' - the value can either be None or must be a particular type
            'list_of_type_none': the value can either be None or must be a particular type or a list of values of a particular type
            'type_constrained' - the value must be a particular type and fulfill given constraints
            'type_constrained_none' - the value can either be None or must be a particular type and fulfill given constraints
            'dict_of_type' - the value must be a dictionary where the values are of a particular type
            'dict_of_type_none' - the value can either be None or must be a dictionary where the values are of a particular type
            'array_of_type' - the value must be an array of values of a particular datatype
            'array_of_type_none' - the value can either be None or must be an array of values of a particular datatype
            'array_of_type_constrained' - the value must be an array of values of a particular datatype and fulfill given constraints
            'array_of_type_constrained_none' - the value can either be None or must be an array of values of a particular datatype and fulfill given constraints
            'set' - the value must be a member of a given set of values
            'any' - the value can be anything
    '''
    def __init__(self, default_dict):
        self.__data = {k: v[0] for k, v in default_dict.items()}
        self.__options = {k: v[1:] for k, v in default_dict.items()}

    def __len__(self):
        return len(self.__data)

    def __iter__(self):
        return iter(self.__data)

    def __setitem__(self, k, v):
        if k not in self.__data:
            raise KeyError(f'Cannot add key {k!r}, as ParamsDict keys are fixed.')

        options_type, options, _, constraints = self.__options[k]
        if options_type == 'type':
            if _is_subtype(v, options):
                self.__data[k] = v
            else:
                raise ValueError(f'Value associated with key {k!r} must be of type {options!r}, but input is {v!r} which is of type {type(v)!r}.')
        elif options_type == 'list_of_type':
            for sub_v in to_list(v):
                if not _is_subtype(sub_v, options):
                    raise ValueError(f'Value associated with key {k!r} must be of type {options!r}, but input is {sub_v!r} which is of type {type(sub_v)!r}.')
            self.__data[k] = v
        elif options_type == 'type_none':
            if (v is None) or _is_subtype(v, options):
                self.__data[k] = v
            else:
                raise ValueError(f'Value associated with key {k!r} must be of type {options!r} or None, but input is {v!r} which is of type {type(v)!r}.',)
        elif options_type == 'list_of_type_none':
            if v is None:
                self.__data[k] = []
            for sub_v in to_list(v):
                if not _is_subtype(sub_v, options):
                    raise ValueError(f'Value associated with key {k!r} must be of type {options!r}, but input is {sub_v!r} which is of type {type(sub_v)!r}.')
            self.__data[k] = v
        elif options_type == 'type_constrained':
            if _is_subtype(v, options[0]):
                if options[1](v):
                    self.__data[k] = v
                else:
                    raise ValueError(f'Value associated with key {k!r} must fulfill the constraint(s) {constraints!r}, but input is {v!r} which does not.')
            elif options[1](v):
                raise ValueError(f'Value associated with key {k!r} must be of type {options[0]!r}, but input is {v!r} which is of type {type(v)!r}.')
            else:
                raise ValueError(f'Value associated with key {k!r} must be of type {options[0]!r}, but input is {v!r} which is of type {type(v)!r}. In addition, the input does not fulfill the constraint(s) {constraints!r}.')
        elif options_type == 'type_constrained_none':
            if v is None:
                self.__data[k] = v
            else:
                if _is_subtype(v, options[0]):
                    if options[1](v):
                        self.__data[k] = v
                    else:
                        raise ValueError(f'Value associated with key {k!r} must be None or fulfill the constraint(s) {constraints!r}, but input is {v!r} which does not.')
                elif options[1](v):
                    raise ValueError(f'Value associated with key {k!r} must be of type {options[0]!r} or None, but input is {v!r} which is of type {type(v)!r}.')
                else:
                    raise ValueError(f'Value associated with key {k!r} must be of type {options[0]!r} or None, but input is {v!r} which is of type {type(v)!r}. In addition, the input does not fulfill the constraint(s) {constraints!r}.')
        elif options_type == 'dict_of_type':
            if _is_subtype(v, dict):
                for sub_k, sub_v in v.items():
                    if not _is_subtype(sub_v, options):
                        raise ValueError(f'Value associated with key {k!r} must be a dictionary with values of type {options!r}, but key {sub_k!r} has associated value {sub_v!r}, which has type {type(sub_v)!r}.')
                self.__data[k] = v
            else:
                raise ValueError(f'Value associated with key {k!r} must be an array, but input is {v!r} which is of type {type(v)!r}.')
        elif options_type == 'dict_of_type_none':
            if v is None:
                self.__data[k] = v
            else:
                if _is_subtype(v, dict):
                    for sub_k, sub_v in v.items():
                        if not _is_subtype(sub_v, options):
                            raise ValueError(f'Value associated with key {k!r} must be a dictionary with values of type {options!r}, but key {sub_k!r} has associated value {sub_v!r}, which has type {type(sub_v)!r}.')
                    self.__data[k] = v
                else:
                    raise ValueError(f'Value associated with key {k!r} must be an array, but input is {v!r} which is of type {type(v)!r}.')
        elif options_type == 'array_of_type':
            if _is_subtype(v, np.ndarray):
                if _is_subdtype(v, options):
                    self.__data[k] = v
                else:
                    raise ValueError(f'Value associated with key {k!r} must be an array of datatype {options!r}, but input is {v!r} which has datatype {v.dtype!r}.')
            else:
                raise ValueError(f'Value associated with key {k!r} must be an array, but input is {v!r} which is of type {type(v)!r}.')
        elif options_type == 'array_of_type_none':
            if v is None:
                self.__data[k] = v
            else:
                if _is_subtype(v, np.ndarray):
                    if _is_subdtype(v, options):
                        self.__data[k] = v
                    else:
                        raise ValueError(f'Value associated with key {k!r} must be None or an array of datatype {options!r}, but input is {v!r} which has datatype {v.dtype!r}.')
                else:
                    raise ValueError(f'Value associated with key {k!r} must be None or an array, but input is {v!r} which is of type {type(v)!r}.')
        elif options_type == 'array_of_type_constrained':
            if _is_subtype(v, np.ndarray):
                if _is_subdtype(v, options[0]):
                    if options[1](v):
                        self.__data[k] = v
                    else:
                        raise ValueError(f'Value associated with key {k!r} must fulfill the constraint(s) {constraints!r}, but input is {v!r} which does not.')
                else:
                    raise ValueError(f'Value associated with key {k!r} must be an array of datatype {options!r}, but input is {v!r} which has datatype {v.dtype!r}.')
            elif options[1](v):
                raise ValueError(f'Value associated with key {k!r} must be an array, but input is {v!r} which is of type {type(v)!r}.')
            else:
                raise ValueError(f'Value associated with key {k!r} must be an array, but input is {v!r} which is of type {type(v)!r}. In addition, the input does not fulfill the constraint(s) {constraints!r}.')
        elif options_type == 'array_of_type_constrained_none':
            if v is None:
                self.__data[k] = v
            else:
                if _is_subtype(v, np.ndarray):
                    if _is_subdtype(v, options[0]):
                        if options[1](v):
                            self.__data[k] = v
                        else:
                            raise ValueError(f'Value associated with key {k!r} must be None or fulfill the constraint(s) {constraints!r}, but input is {v!r} which does not.')
                    else:
                        raise ValueError(f'Value associated with key {k!r} must be None or an array of datatype {options!r}, but input is {v!r} which has datatype {v.dtype!r}.')
                elif options[1](v):
                    raise ValueError(f'Value associated with key {k!r} must be None or an array, but input is {v!r} which is of type {type(v)!r}.')
                else:
                    raise ValueError(f'Value associated with key {k!r} must be None or an array, but input is {v!r} which is of type {type(v)!r}. In addition, the input does not fulfill the constraint(s) {constraints!r}.')
        elif options_type == 'set':
            if v in to_list(options):
                self.__data[k] = v
            else:
                raise ValueError(f'Value associated with key {k!r} must be a subset of {options!r}, but input is {v!r}.')
        elif options_type == 'any':
            self.__data[k] = v
        else:
            raise NotImplementedError('Invalid options type')

    def __delitem__(self, k):
        raise NotImplementedError

    def __getitem__(self, k):
        return self.__data[k]

    def __contains__(self, k):
        return k in self.__data

    def __repr__(self):
        return dict(self).__repr__()

    def keys(self):
        return self.__data.keys()

    def values(self):
        return self.__data.values()

    def items(self):
        return self.__data.items()

    def copy(self):
        data_copy = self.__data.copy()
        options_copy = self.__options.copy()
        return ParamsDict({k: (v, *options_copy[k]) for k, v in data_copy.items()})

    def get_multiple(self, ks):
        '''
        Access the values associated with multiple keys.

        Arguments:
            ks (immutable or list of immutable): key(s) to access values

        Returns
            (tuple): value(s) associated with key(s)
        '''
        return (self[k] for k in to_list(ks))

    def describe(self, k):
        '''
        Describe what a particular key-value pair does.

        Arguments:
            k (immutable): key
        '''
        options_type, options, description, constraints = self.__options[k]
        print(f'KEY: {k!r}')
        print(f'CURRENT VALUE: {self[k]!r}')
        if options_type == 'type':
            print(f'VALID VALUES: one of type {options!r}')
        elif options_type == 'list_of_type':
            print(f'VALID VALUES: one of or list of type {options!r}')
        elif options_type == 'type_none':
            print(f'VALID VALUES: None or one of type {options!r}')
        elif options_type == 'list_of_type_none':
            print(f'VALID VALUES: None or one of or list of type {options!r}')
        elif options_type == 'type_constrained':
            print(f'VALID VALUES: one of type {options[0]!r}')
            print(f'CONSTRAINTS: {constraints!r}')
        elif options_type == 'type_constrained_none':
            print(f'VALID VALUES: None or one of type {options[0]!r}')
            print(f'CONSTRAINTS: {constraints!r}')
        elif options_type == 'dict_of_type':
            print(f'VALID VALUES: dictionary of type {options!r}')
        elif options_type == 'dict_of_type_none':
            print(f'VALID VALUES: None or dictionary of type {options!r}')
        elif options_type == 'array_of_type':
            print(f'VALID VALUES: array of datatype {options!r}')
        elif options_type == 'array_of_type_none':
            print(f'VALID VALUES: None or array of datatype {options!r}')
        elif options_type == 'array_of_type_constrained':
            print(f'VALID VALUES: array of datatype {options!r}')
            print(f'CONSTRAINTS: {constraints!r}')
        elif options_type == 'array_of_type_constrained_none':
            print(f'VALID VALUES: None or array of datatype {options!r}')
            print(f'CONSTRAINTS: {constraints!r}')
        elif options_type == 'set':
            print(f'VALID VALUES: one of {options!r}')
        elif options_type == 'any':
            print('VALID VALUES: anything')
        print(f'DESCRIPTION: {description}')

    def describe_all(self):
        '''
        Describe all key-value pairs.
        '''
        for k in self.keys():
            self.describe(k)

def update_dict(default_params, user_params):
    '''
    Replace entries in default_params with values in user_params. This function allows user_params to include only a subset of the required parameters in the dictionary.

    Arguments:
        default_params (dict): default parameter values
        user_params (dict): user selected parameter values

    Returns:
        (dict): default_params updated with parameter values in user_params
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
    if not isinstance(data, (list, tuple, set, frozenset)):
        return [data]
    return list(data)

def fast_shift(arr, num, fill_value=np.nan):
    '''
    Shift array by a given number of elements, filling values rolled around with fill_value. Source: https://stackoverflow.com/a/42642326/17333120.

    Arguments:
        arr (NumPy Array): array to shift
        num (int): how many elements to shift
        fill_value (any): value to input for elements that roll around

    Returns:
        (NumPy Array): shifted array
    '''
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result[:] = arr
    return result

class ChainedAssignment:
    '''
    Context manager to temporarily set pandas chained assignment warning. Source: https://stackoverflow.com/a/53954986/17333120. Usage:
        with ChainedAssignment():
            --code with no warnings--
        with ChainedAssignment('error'):
            --code with errors--
    '''
    def __init__(self, chained=None):
        acceptable = [None, 'warn', 'raise']
        assert chained in acceptable, "chained must be in " + str(acceptable)
        self.swcw = chained

    def __enter__(self):
        self.saved_swcw = pd.options.mode.chained_assignment
        pd.options.mode.chained_assignment = self.swcw
        return self

    def __exit__(self, *args):
        pd.options.mode.chained_assignment = self.saved_swcw

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
        Path(f'logs/{obj_name}_logs').mkdir(parents=True, exist_ok=True)
        # Create file handler which logs even debug messages
        fh = logging.FileHandler(f'logs/{obj_name}_logs/{obj_name}_spam.log')
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
    col1 = frame.loc[:, col_groupby].to_numpy()
    col2 = frame.loc[:, col_grouped].to_numpy()
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

def compare_frames(frame1, frame2, size_variable='len', operator='geq', save_to_frame1=False, is_sorted=False):
    '''
    Compare two frames using a particular size property and operator.

    Arguments:
        frame1 (BipartiteBase): first frame
        frame2 (BipartiteBase): second frame
        size_variable (str): what size variable to use to compare frames. Options are 'len'/'length' (length of frames), 'firms' (number of unique firms), 'workers' (number of unique workers), 'stayers' (number of unique stayers), 'movers' (number of unique movers), 'firms_plus_workers' (number of unique firms + number of unique workers), 'firms_plus_stayers' (number of unique firms + number of unique stayers), 'firms_plus_movers' (number of unique firms + number of unique movers), 'len_stayers'/'length_stayers' (number of stayer observations), 'len_movers'/'length_movers' (number of mover observations), 'stays' (number of stay observations), and 'moves' (number of move observations).
        operator (str): how to compare properties. Options are 'eq' (equality), 'gt' (greater than), 'lt' (less than), 'geq' (greater than or equal to), and 'leq' (less than or equal to).
        save_to_frame1 (bool): if True, save size_variable for frame1 to attribute frame1.comp_size
        is_sorted (bool): if False, dataframe will be sorted by i in a groupby (but self will not be not sorted). Set to True if dataframe is already sorted.
    '''
    # First, get the values for the frames corresponding to the given property
    property_dict = {
        'len': lambda a: len(a),
        'length': lambda a: len(a),
        'firms': lambda a: a.n_firms(),
        'workers': lambda a: a.n_workers(),
        'stayers': lambda a: a.loc[a.loc[:, 'm'].to_numpy() == 0, :].n_unique_ids('i'),
        'movers': lambda a: a.loc[a.loc[:, 'm'].to_numpy() > 0, :].n_unique_ids('i'),
        'firms_plus_workers': lambda a: a.n_firms() + a.n_workers(),
        'firms_plus_stayers': lambda a: a.n_firms() + a.loc[a.loc[:, 'm'].to_numpy() == 0, :].n_unique_ids('i'),
        'firms_plus_movers': lambda a: a.n_firms() + a.loc[a.loc[:, 'm'].to_numpy() > 0, :].n_unique_ids('i'),
        'len_stayers': lambda a: len(a.loc[~(a.get_worker_m(is_sorted)), :]),
        'length_stayers': lambda a: len(a.loc[~(a.get_worker_m(is_sorted)), :]),
        'len_movers': lambda a: len(a.loc[a.get_worker_m(is_sorted), :]),
        'length_movers': lambda a: len(a.loc[a.get_worker_m(is_sorted), :]),
        'stays': lambda a: len(a.loc[a.loc[:, 'm'].to_numpy() == 0, :]),
        'moves': lambda a: len(a.loc[a.loc[:, 'm'].to_numpy() > 0, :])
    }
    try:
        val1 = frame1.comp_size
    except AttributeError:
        val1 = property_dict[size_variable](frame1)
        if save_to_frame1:
            frame1.comp_size = val1
    val2 = property_dict[size_variable](frame2)

    # Second, compare the values using the given operator
    operator_dict = {
        'eq': lambda a, b: a == b,
        'gt': lambda a, b: a > b,
        'lt': lambda a, b: a < b,
        'geq': lambda a, b: a >= b,
        'leq': lambda a, b: a <= b
    }

    return operator_dict[operator](val1, val2)
