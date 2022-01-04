'''
Class for a bipartite network
'''
from pandas.core.indexes.base import InvalidIndexError
from tqdm.auto import tqdm
import numpy as np
# from numpy_groupies.aggregate_numpy import aggregate
import pandas as pd
from pandas import DataFrame, Int64Dtype
# from scipy.sparse.csgraph import connected_components
import warnings
import bipartitepandas as bpd
from bipartitepandas import col_order, update_dict, to_list, logger_init, col_dict_optional_cols, aggregate_transform
import igraph as ig

def recollapse_loop(force=False):
    '''
    Decorator function that accounts for issues with selecting ids under particular restrictions for collapsed data. In particular, looking at a restricted set of observations can require recollapsing data, which can they change which observations meet the given restrictions. This function loops until stability is achieved.

    Arguments:
        force (bool): if True, force loop for non-collapsed data
    '''
    def recollapse_loop_inner(func):
        def recollapse_loop_inner_inner(*args, **kwargs):
            # Do function
            self = args[0]
            frame = func(*args, **kwargs)

            if force or isinstance(self, (bpd.BipartiteLongCollapsed, bpd.BipartiteEventStudyCollapsed)):
                kwargs['copy'] = False
                if len(frame) != len(self):
                    # If the frame changes, we have to re-loop until stability
                    frame_prev = frame
                    frame = func(frame_prev, *args[1:], **kwargs)
                    while len(frame) != len(frame_prev):
                        frame_prev = frame
                        frame = func(frame_prev, *args[1:], **kwargs)

            return frame
        return recollapse_loop_inner_inner
    return recollapse_loop_inner

class BipartiteBase(DataFrame):
    '''
    Base class for BipartitePandas, where BipartitePandas gives a bipartite network of firms and workers. Contains generalized methods. Inherits from DataFrame.

    Arguments:
        *args: arguments for Pandas DataFrame
        columns_req (list): required columns (only put general column names for joint columns, e.g. put 'j' instead of 'j1', 'j2'; then put the joint columns in reference_dict)
        columns_opt (list): optional columns (only put general column names for joint columns, e.g. put 'g' instead of 'g1', 'g2'; then put the joint columns in reference_dict)
        columns_contig (dictionary): columns requiring contiguous ids linked to boolean of whether those ids are contiguous, or None if column(s) not included, e.g. {'i': False, 'j': False, 'g': None} (only put general column names for joint columns)
        reference_dict (dict): clarify which columns are associated with a general column name, e.g. {'i': 'i', 'j': ['j1', 'j2']}
        col_dtype_dict (dict): link column to datatype
        col_dict (dict or None): make data columns readable. Keep None if column names already correct
        include_id_reference_dict (bool): if True, create dictionary of Pandas dataframes linking original id values to contiguous id values
        **kwargs: keyword arguments for Pandas DataFrame
    '''
    _metadata = ['col_dict', 'reference_dict', 'id_reference_dict', 'col_dtype_dict', 'columns_req', 'columns_opt', 'columns_contig', 'default_cluster', 'dtype_dict', 'default_clean', 'connected', 'correct_cols', 'no_na', 'no_duplicates', 'i_t_unique'] # Attributes, required for Pandas inheritance

    def __init__(self, *args, columns_req=[], columns_opt=[], columns_contig=[], reference_dict={}, col_dtype_dict={}, col_dict=None, include_id_reference_dict=False, **kwargs):
        # Initialize DataFrame
        super().__init__(*args, **kwargs)

        # Start logger
        logger_init(self)
        # self.logger.info('initializing BipartiteBase object')

        if len(args) > 0 and isinstance(args[0], BipartiteBase): # Note that isinstance works for subclasses
            self._set_attributes(args[0], include_id_reference_dict)
        else:
            self.columns_req = ['i', 'j', 'y'] + columns_req
            self.columns_opt = ['g', 'm'] + columns_opt
            self.columns_contig = update_dict({'i': False, 'j': False, 'g': None}, columns_contig)
            self.reference_dict = update_dict({'i': 'i', 'm': 'm'}, reference_dict)
            self._reset_id_reference_dict(include_id_reference_dict) # Link original id values to contiguous id values
            self.col_dtype_dict = update_dict({'i': 'int', 'j': 'int', 'y': 'float', 't': 'int', 'g': 'int', 'm': 'int'}, col_dtype_dict)
            default_col_dict = {}
            for col in to_list(self.columns_req):
                for subcol in to_list(self.reference_dict[col]):
                    default_col_dict[subcol] = subcol
            for col in to_list(self.columns_opt):
                for subcol in to_list(self.reference_dict[col]):
                    default_col_dict[subcol] = None

            # Create self.col_dict
            self.col_dict = col_dict_optional_cols(default_col_dict, col_dict, self.columns, optional_cols=[self.reference_dict[col] for col in self.columns_opt])

            # Set attributes
            self._reset_attributes()

        self.default_cluster = {
            'measure_cdf': False, # If True, approximate firm-level income cdfs
            'measure_moments': False, # If True, approximate firm-level moments
            'cluster_KMeans': False, # If True, cluster using KMeans. Valid only if cluster_quantiles=False.
            'cluster_quantiles': False # If True, cluster using quantiles. Valid only if cluster_KMeans=False, measure_cdf=False, and measure_moments=True, and measuring only a single moment.
        }

        self.dtype_dict = {
            'int': ['int', 'int8', 'int16', 'int32', 'int64', 'Int64'],
            'float': ['float', 'float8', 'float16', 'float32', 'float64', 'float128', 'int', 'int8', 'int16', 'int32', 'int64', 'Int64'],
            'str': 'str'
        }

        self.default_clean = {
            'connectedness': 'connected', # When computing largest connected set of firms: if 'connected', keep observations in the largest connected set of firms; if 'biconnected_firms' or 'biconnected_observations', keep observations in the largest biconnected set of firms; if None, keep all observations
            'i_t_how': 'max', # When dropping i-t duplicates: if 'max', keep max paying job; if 'sum', sum over duplicate worker-firm-year observations, then take the highest paying worker-firm sum; if 'mean', average over duplicate worker-firm-year observations, then take the highest paying worker-firm average. Note that if multiple time and/or firm columns are included (as in event study format), then data is converted to long, cleaned, then reconverted to its original format
            'drop_multiples': False, # If True, rather than collapsing over spells, drop any spells with multiple observations (this is for computational efficiency when re-collapsing data for biconnected components)
            'data_validity': True, # If True, run data validity checks; much faster if set to False
            'copy': False # If False, avoid copying data when possible
        }

        # self.logger.info('BipartiteBase object initialized')

    @property
    def _constructor(self):
        '''
        For inheritance from Pandas.
        '''
        return BipartiteBase

    def copy(self):
        '''
        Copy.

        Returns:
            bdf_copy (BipartiteBase): copy of instance
        '''
        df_copy = DataFrame(self, copy=True)
        bdf_copy = self._constructor(df_copy)
        bdf_copy._set_attributes(self) # This copies attribute dictionaries, default copy does not

        return bdf_copy

    def summary(self):
        '''
        Print summary statistics.
        '''
        mean_wage = np.mean(self.loc[:, self.reference_dict['y']].to_numpy())
        max_wage = np.max(self.loc[:, self.reference_dict['y']].to_numpy())
        min_wage = np.min(self.loc[:, self.reference_dict['y']].to_numpy())
        ret_str = 'format: ' + type(self).__name__ + '\n'
        ret_str += 'number of workers: ' + str(self.n_workers()) + '\n'
        ret_str += 'number of firms: ' + str(self.n_firms()) + '\n'
        ret_str += 'number of observations: ' + str(len(self)) + '\n'
        ret_str += 'mean wage: ' + str(mean_wage) + '\n'
        ret_str += 'max wage: ' + str(max_wage) + '\n'
        ret_str += 'min wage: ' + str(min_wage) + '\n'
        ret_str += 'connected (None if not ignoring connected set): ' + str(self.connected) + '\n'
        for contig_col, is_contig in self.columns_contig.items():
            ret_str += 'contiguous {} ids (None if not included): '.format(contig_col) + str(is_contig) + '\n'
        ret_str += 'correct column names and types: ' + str(self.correct_cols) + '\n'
        ret_str += 'no nans: ' + str(self.no_na) + '\n'
        ret_str += 'no duplicates: ' + str(self.no_duplicates) + '\n'
        ret_str += 'i-t (worker-year) observations unique (None if t column(s) not included): ' + str(self.i_t_unique) + '\n'

        print(ret_str)

    def unique_ids(self, id_col):
        '''
        Unique ids in column.

        Arguments:
            id_col (str): column to check ids ('i', 'j', or 'g'). Use general column names for joint columns, e.g. put 'j' instead of 'j1', 'j2'

        Returns:
            (NumPy Array): unique ids
        '''
        id_lst = []
        for id_subcol in to_list(self.reference_dict[id_col]):
            id_lst += list(self.loc[:, id_subcol].unique())
        return np.array(list(set(id_lst)))

    def n_unique_ids(self, id_col):
        '''
        Number of unique ids in column.

        Arguments:
            id_col (str): column to check ids ('i', 'j', or 'g'). Use general column names for joint columns, e.g. put 'j' instead of 'j1', 'j2'

        Returns:
            (int): number of unique ids
        '''
        return len(self.unique_ids(id_col))

    def n_workers(self):
        '''
        Get the number of unique workers.

        Returns:
            (int): number of unique workers
        '''
        return self.loc[:, 'i'].nunique()

    def n_firms(self):
        '''
        Get the number of unique firms.

        Returns:
            (int): number of unique firms
        '''
        return self.n_unique_ids('j')

    def n_clusters(self):
        '''
        Get the number of unique clusters.

        Returns:
            (int or None): number of unique clusters, None if not clustered
        '''
        if not self._col_included('g'): # If cluster column not in dataframe
            return None
        return self.n_unique_ids('g')

    def original_ids(self, copy=True):
        '''
        Return self merged with original column ids.

        Arguments:
            copy (bool): if False, avoid copy

        Returns:
            (BipartiteBase or None): copy of self merged with original column ids, or None if id_reference_dict is empty
        '''
        frame = pd.DataFrame(self, copy=copy)
        if self.id_reference_dict:
            for id_col, reference_df in self.id_reference_dict.items():
                if len(reference_df) > 0: # Make sure non-empty
                    for id_subcol in to_list(self.reference_dict[id_col]):
                        try:
                            frame = frame.merge(reference_df.loc[:, ['original_ids', 'adjusted_ids_' + str(len(reference_df.columns) - 1)]].rename({'original_ids': 'original_' + id_subcol, 'adjusted_ids_' + str(len(reference_df.columns) - 1): id_subcol}, axis=1), how='left', on=id_subcol)
                        except TypeError: # Int64 error with NaNs
                            frame.loc[:, id_col] = frame.loc[:, id_col].astype('Int64', copy=False)
                            frame = frame.merge(reference_df.loc[:, ['original_ids', 'adjusted_ids_' + str(len(reference_df.columns) - 1)]].rename({'original_ids': 'original_' + id_subcol, 'adjusted_ids_' + str(len(reference_df.columns) - 1): id_subcol}, axis=1), how='left', on=id_subcol)
                # else:
                #     # If no changes, just make original_id be the same as the current id
                #     for id_subcol in to_list(self.reference_dict[id_col]):
                #         frame['original_' + id_subcol] = frame[id_subcol]
            return frame
        else:
            warnings.warn('id_reference_dict is empty. Either your id columns are already correct, or you did not specify `include_id_reference_dict=True` when initializing your BipartitePandas object')

            return None

    def _set_attributes(self, frame, no_dict=False, include_id_reference_dict=False):
        '''
        Set class attributes to equal those of another BipartitePandas object.

        Arguments:
            frame (BipartitePandas): BipartitePandas object whose attributes to use
            no_dict (bool): if True, only set booleans, no dictionaries
            include_id_reference_dict (bool): if True, create dictionary of Pandas dataframes linking original id values to contiguous id values
        '''
        # Dictionaries
        if not no_dict:
            self.columns_req = frame.columns_req.copy()
            self.columns_opt = frame.columns_opt.copy()
            self.reference_dict = frame.reference_dict.copy()
            self.col_dtype_dict = frame.col_dtype_dict.copy()
            self.col_dict = frame.col_dict.copy()
        self.columns_contig = frame.columns_contig.copy() # Required, even if no_dict
        if frame.id_reference_dict:
            self.id_reference_dict = {}
            # Must do a deep copy
            for id_col, reference_df in frame.id_reference_dict.items():
                self.id_reference_dict[id_col] = reference_df.copy()
        else:
            # This is if the original dataframe DIDN'T have an id_reference_dict (but the new dataframe may or may not)
            self._reset_id_reference_dict(include_id_reference_dict)
        # # Logger
        # self.logger = frame.logger
        # Booleans
        self.connected = frame.connected # If False, not connected; if 'connected', all observations are in the largest connected set of firms; if 'biconnected' all observations are in the largest biconnected set of firms; if None, connectedness ignored
        self.correct_cols = frame.correct_cols # If True, column names are correct
        self.no_na = frame.no_na # If True, no NaN observations in the data
        self.no_duplicates = frame.no_duplicates # If True, no duplicate rows in the data
        self.i_t_unique = frame.i_t_unique # If True, each worker has at most one observation per period

    def _reset_attributes(self, columns_contig=True, connected=True, correct_cols=True, no_na=True, no_duplicates=True, i_t_unique=True):
        '''
        Reset class attributes conditions to be False/None.

        Arguments:
            columns_contig (bool): if True, reset self.columns_contig
            connected (bool): if True, reset self.connected
            correct_cols (bool): if True, reset self.correct_cols
            no_na (bool): if True, reset self.no_na
            no_duplicates (bool): if True, reset self.no_duplicates
            i_t_unique (bool): if True, reset self.i_t_unique

        Returns:
            self (BipartiteBase): self with reset class attributes
        '''
        if columns_contig:
            for contig_col in self.columns_contig.keys():
                if self._col_included(contig_col):
                    self.columns_contig[contig_col] = False
                else:
                    self.columns_contig[contig_col] = None
        if connected:
            self.connected = None # If False, not connected; if 'connected', all observations are in the largest connected set of firms; if 'biconnected' all observations are in the largest biconnected set of firms; if None, connectedness ignored
        if correct_cols:
            self.correct_cols = False # If True, column names are correct
        if no_na:
            self.no_na = False # If True, no NaN observations in the data
        if no_duplicates:
            self.no_duplicates = False # If True, no duplicate rows in the data
        if i_t_unique:
            self.i_t_unique = None # If True, each worker has at most one observation per period; if None, t column not included (set to False later in method if t column included)

            # Verify whether period included
            if self._col_included('t'):
                self.i_t_unique = False

        # logger_init(self)

        return self

    def _reset_id_reference_dict(self, include=False):
        '''
        Reset id_reference_dict.

        Arguments:
            include (bool): if True, id_reference_dict will track changes in ids

        Returns:
            self (BipartiteBase): self with reset id_reference_dict
        '''
        if include:
            self.id_reference_dict = {id_col: pd.DataFrame() for id_col in self.reference_dict.keys()}
        else:
            self.id_reference_dict = {}

        return self

    def _col_included(self, col):
        '''
        Check whether a column from the pre-established required/optional lists is included.

        Arguments:
            col (str): column to check. Use general column names for joint columns, e.g. put 'j' instead of 'j1', 'j2'

        Returns:
            (bool): if True, column is included
        '''
        if col in self.columns_req + self.columns_opt:
            for subcol in to_list(self.reference_dict[col]):
                if self.col_dict[subcol] is None:
                    return False
            return True
        return False

    def _included_cols(self, flat=False):
        '''
        Get all columns included from the pre-established required/optional lists.
        
        Arguments:
            flat (bool): if False, uses general column names for joint columns, e.g. returns 'j' instead of 'j1', 'j2'.

        Returns:
            all_cols (list): included columns
        '''
        all_cols = []
        for col in self.columns_req + self.columns_opt:
            include = True
            for subcol in to_list(self.reference_dict[col]):
                if self.col_dict[subcol] is None:
                    include = False
                    break
            if include:
                if flat:
                    all_cols += to_list(self.reference_dict[col])
                else:
                    all_cols.append(col)
        return all_cols

    def drop(self, indices, axis=1, inplace=True, allow_required=False):
        '''
        Drop indices along axis.

        Arguments:
            indices (int or str, optionally as a list): row(s) or column(s) to drop. For columns, use general column names for joint columns, e.g. put 'g' instead of 'g1', 'g2'. Only optional columns may be dropped
            axis (int): 0 to drop rows, 1 to drop columns
            inplace (bool): if True, modify in-place
            allow_required (bool): if True, allow to drop required columns

        Returns:
            frame (BipartiteBase): BipartiteBase with dropped indices
        '''
        if inplace:
            frame = self
        else:
            frame = self.copy()

        if axis == 1:
            for col in to_list(indices):
                if col in frame.columns or col in frame.columns_req or col in frame.columns_opt:
                    if col in frame.columns_opt: # If column optional
                        for subcol in to_list(frame.reference_dict[col]):
                            DataFrame.drop(frame, subcol, axis=1, inplace=True)
                            frame.col_dict[subcol] = None
                        if col in frame.columns_contig.keys(): # If column contiguous
                            frame.columns_contig[col] = None
                            if frame.id_reference_dict: # If id_reference_dict has been initialized
                                frame.id_reference_dict[col] = pd.DataFrame()
                    elif col not in frame._included_cols() and col not in frame._included_cols(flat=True): # If column is not pre-established
                        DataFrame.drop(frame, col, axis=1, inplace=True)
                    else:
                        if not allow_required:
                            warnings.warn("{} is either (a) a required column and cannot be dropped or (b) a subcolumn that can be dropped, but only by specifying the general column name (e.g. use 'g' instead of 'g1' or 'g2')".format(col))
                        else:
                            DataFrame.drop(frame, col, axis=1, inplace=True)
                else:
                    warnings.warn('{} is not in data columns'.format(col))
        elif axis == 0:
            DataFrame.drop(frame, indices, axis=0, inplace=True)
            frame._reset_attributes()
            frame.clean_data({'connectedness': frame.connected})

        return frame

    def rename(self, rename_dict, inplace=True):
        '''
        Rename a column.

        Arguments:
            rename_dict (dict): key is current column name, value is new column name. Use general column names for joint columns, e.g. put 'g' instead of 'g1', 'g2'. Only optional columns may be renamed
            inplace (bool): if True, modify in-place

        Returns:
            frame (BipartiteBase): BipartiteBase with renamed columns
        '''
        if inplace:
            frame = self
        else:
            frame = self.copy()

        for col_cur, col_new in rename_dict.items():
            if col_cur in frame.columns or col_cur in frame.columns_req or col_cur in frame.columns_opt:
                if col_cur in self.columns_opt: # If column optional
                    if len(to_list(self.reference_dict[col_cur])) > 1:
                        for i, subcol in enumerate(to_list(self.reference_dict[col_cur])):
                            DataFrame.rename(frame, {subcol: col_new + str(i + 1)}, axis=1, inplace=True)
                            frame.col_dict[subcol] = None
                    else:
                        DataFrame.rename(frame, {col_cur: col_new}, axis=1, inplace=True)
                        frame.col_dict[col_cur] = None
                    if col_cur in frame.columns_contig.keys(): # If column contiguous
                            frame.columns_contig[col_cur] = None
                            if frame.id_reference_dict: # If id_reference_dict has been initialized
                                frame.id_reference_dict[col_cur] = pd.DataFrame()
                elif col_cur not in frame._included_cols() and col_cur not in frame._included_cols(flat=True): # If column is not pre-established
                        DataFrame.rename(frame, {col_cur: col_new}, axis=1, inplace=True)
                else:
                    warnings.warn("{} is either (a) a required column and cannot be renamed or (b) a subcolumn that can be renamed, but only by specifying the general column name (e.g. use 'g' instead of 'g1' or 'g2')".format(col_cur))
            else:
                warnings.warn('{} is not in data columns'.format(col_cur))

        return frame

    def merge(self, *args, **kwargs):
        '''
        Merge two BipartiteBase objects.

        Arguments:
            *args: arguments for Pandas merge
            **kwargs: keyword arguments for Pandas merge

        Returns:
            frame (BipartiteBase): merged dataframe
        '''
        frame = DataFrame.merge(self, *args, **kwargs)
        frame = self._constructor(frame) # Use correct constructor
        if kwargs['how'] == 'left': # Non-left merge could cause issues with data, by default resets attributes
            frame._set_attributes(self)
        return frame

    def _contiguous_ids(self, id_col, copy=True):
        '''
        Make column of ids contiguous.

        Arguments:
            id_col (str): column to make contiguous ('i', 'j', or 'g'). Use general column names for joint columns, e.g. put 'j' instead of 'j1', 'j2'. Only optional columns may be renamed
            copy (bool): if False, avoid copy

        Returns:
            frame (BipartiteBase): BipartiteBase with contiguous ids
        '''
        if copy:
            frame = self.copy()
        else:
            frame = self

        cols = to_list(frame.reference_dict[id_col])
        n_cols = len(cols)
        n_rows = len(frame)
        all_ids = frame.loc[:, cols].to_numpy().reshape(n_cols * n_rows)
        # Source: https://stackoverflow.com/questions/16453465/multi-column-factorize-in-pandas
        factorized = pd.factorize(all_ids)

        # Quickly check whether ids need to be reset
        try:
            if max(factorized[1]) + 1 == len(factorized[1]):
                return frame
        except TypeError:
            # If ids are not integers, this will return a TypeError and we can ignore it
            pass

        frame.loc[:, cols] = factorized[0].reshape((n_rows, n_cols))

        # Save id reference dataframe, so user can revert back to original ids
        if frame.id_reference_dict: # If id_reference_dict has been initialized
            if len(frame.id_reference_dict[id_col]) == 0: # If dataframe empty, start with original ids: adjusted ids
                frame.id_reference_dict[id_col].loc[:, 'original_ids'] = factorized[1]
                frame.id_reference_dict[id_col].loc[:, 'adjusted_ids_1'] = np.arange(len(factorized[1]))
            else: # Merge in new adjustment step
                n_cols_id = len(frame.id_reference_dict[id_col].columns)
                id_reference_df = pd.DataFrame({'adjusted_ids_' + str(n_cols_id - 1): factorized[1], 'adjusted_ids_' + str(n_cols_id): np.arange(len(factorized[1]))}, index=np.arange(len(factorized[1]))).astype('Int64', copy=False)
                frame.id_reference_dict[id_col] = frame.id_reference_dict[id_col].merge(id_reference_df, how='left', on='adjusted_ids_' + str(n_cols_id - 1))

        # Sort columns
        frame = frame.sort_cols(copy=False)

        # ids are now contiguous
        frame.columns_contig[id_col] = True

        return frame

    def _update_cols(self, inplace=True):
        '''
        Rename columns and keep only relevant columns.

        Arguments:
            inplace (bool): if True, modify in-place

        Returns:
            frame (BipartiteBase): BipartiteBase with updated columns
        '''
        if inplace:
            frame = self
        else:
            frame = self.copy()

        new_col_dict = {}
        rename_dict = {} # For renaming columns in data
        keep_cols = []

        for key, val in frame.col_dict.items():
            if val is not None:
                rename_dict[val] = key
                new_col_dict[key] = key
                keep_cols.append(key)
            else:
                new_col_dict[key] = None
        frame.col_dict = new_col_dict
        keep_cols = sorted(keep_cols, key=col_order) # Sort columns
        DataFrame.rename(frame, rename_dict, axis=1, inplace=True)
        for col in frame.columns:
            if col not in keep_cols:
                frame.drop(col)

        return frame

    def clean_data(self, user_clean={}):
        '''
        Clean data to make sure there are no NaN or duplicate observations, firms are connected by movers and firm ids are contiguous.

        Arguments:
            user_clean (dict): dictionary of parameters for cleaning

                Dictionary parameters:

                    connectedness (str or None, default='connected'): if 'connected', keep observations in the largest connected set of firms; if 'biconnected', keep observations in the largest biconnected set of firms; if None, keep all observations

                    i_t_how (str, default='max'): if 'max', keep max paying job; if 'sum', sum over duplicate worker-firm-year observations, then take the highest paying worker-firm sum; if 'mean', average over duplicate worker-firm-year observations, then take the highest paying worker-firm average. Note that if multiple time and/or firm columns are included (as in event study format), then data is converted to long, cleaned, then reconverted to its original format

                    drop_multiples (bool): if True, rather than collapsing over spells, drop any spells with multiple observations (this is for computational efficiency when re-collapsing data for biconnected components)

                    data_validity (bool, default=True): if True, run data validity checks; much faster if set to False

                    copy (bool, default=False): if False, avoid copy

        Returns:
            frame (BipartiteBase): BipartiteBase with cleaned data
        '''
        self.logger.info('beginning BipartiteBase data cleaning')

        clean_params = update_dict(self.default_clean, user_clean)
        data_validity = clean_params['data_validity']

        if clean_params['copy']:
            frame = self.copy()
        else:
            frame = self

        # First, correct column names
        frame.logger.info('correcting column names')
        frame._update_cols()

        # Next, check that required columns are included and datatypes are correct
        frame.logger.info('checking required columns and datatypes')
        frame = frame._check_cols()

        # Next, sort rows
        frame.logger.info('sorting rows')
        frame.sort_values(['i', to_list(self.reference_dict['t'])[0]], inplace=True)

        # Next, drop NaN observations
        if (not frame.no_na) or data_validity:
            frame.logger.info('dropping NaN observations')
            frame = frame.dropna()

            # Update no_na
            frame.no_na = True

        # Generate 'm' column - this is necessary for the next steps
        # 'm' will get updated in the following steps as it changes
        frame.logger.info("generating 'm' column")
        frame = frame.gen_m(force=True, copy=False)

        # Next, make sure i-t (worker-year) observations are unique
        if frame.i_t_unique is not None and ((not frame.i_t_unique) or data_validity):
            frame.logger.info('keeping highest paying job for i-t (worker-year) duplicates')
            frame = frame._drop_i_t_duplicates(how=clean_params['i_t_how'], copy=False)

            # Update no_duplicates
            frame.no_duplicates = True
        elif (not frame.no_duplicates) or data_validity:
            # Drop duplicate observations
            frame.logger.info('dropping duplicate observations')
            frame.drop_duplicates(inplace=True)

            # Update no_duplicates
            frame.no_duplicates = True

        # Next, check contiguous ids before using igraph (igraph resets ids to be contiguous, so we need to make sure ours are comparable)
        for contig_col, is_contig in frame.columns_contig.items():
            if is_contig is not None and ((not is_contig) or data_validity):
                frame.logger.info('making {} ids contiguous'.format(contig_col))
                frame = frame._contiguous_ids(id_col=contig_col, copy=False)

        # Next, find largest set of firms connected by movers
        if frame.connected in [False, None] or data_validity:
            # Generate largest connected set
            frame.logger.info('generating largest connected set')
            frame = frame._conset(connectedness=clean_params['connectedness'], drop_multiples=clean_params['drop_multiples'], copy=False)

            # Next, check contiguous ids after igraph, in case the connected components dropped ids
            for contig_col, is_contig in frame.columns_contig.items():
                if is_contig is not None and ((not is_contig) or data_validity):
                    frame.logger.info('making {} ids contiguous'.format(contig_col))
                    frame = frame._contiguous_ids(id_col=contig_col, copy=False)

        # Sort columns
        frame.logger.info('sorting columns')
        frame = frame.sort_cols(copy=False)

        # Reset index
        frame.reset_index(drop=True, inplace=True)

        frame.logger.info('BipartiteBase data cleaning complete')

        return frame

    def _check_cols(self):
        '''
        Check that required columns are included, and that all columns have the correct datatype. Raises a ValueError if either is false.

        Returns:
            frame (BipartiteBase): BipartiteBase with contiguous ids, for columns that started with incorrect datatypes
        '''
        frame = self
        cols_included = True
        correct_dtypes = True

        # Check all included columns
        for col in frame._included_cols():
            for subcol in to_list(frame.reference_dict[col]):
                if subcol not in frame.columns:
                    # If column missing
                    frame.logger.info('{} missing from data'.format(subcol))
                    cols_included = False
                else:
                    # If column included, check type
                    col_type = str(frame[subcol].dtype)
                    valid_types = to_list(frame.dtype_dict[frame.col_dtype_dict[col]])
                    if col_type not in valid_types:
                        if col in frame.columns_contig.keys():
                            # If column contiguous, we don't worry about datatype, but will log it
                            frame.logger.info('{} has dtype {}, so we have converted it to contiguous integers'.format(subcol, col_type))
                            frame = frame._contiguous_ids(id_col=col, copy=False)
                        else:
                            frame.logger.info('{} has wrong dtype, it is currently {} but should be one of the following: {}'.format(subcol, col_type, valid_types))
                            correct_dtypes = False

        if (not cols_included) or (not correct_dtypes):
            raise ValueError('Your data does not include the correct columns or column datatypes. The BipartitePandas object cannot be generated with your data. Please see the generated logs for more information.')

        return frame

    def _drop_i_t_duplicates(self, how='max', copy=True):
        '''
        Keep only the highest paying job for i-t (worker-year) duplicates.

        Arguments:
            how (str): if 'max', keep max paying job; otherwise, take `how` over duplicate worker-firm-year observations, then take the highest paying worker-firm observation. `how` can take any option valid for a Pandas transform. Note that if multiple time and/or firm columns are included (as in event study format), then duplicates are cleaned in order of earlier time columns to later time columns, and earlier firm ids to later firm ids
            copy (bool): if False, avoid copy

        Returns:
            frame (BipartiteBase): dataframe that keeps only the highest paying job for i-t (worker-year) duplicates. If no t column(s), returns frame with no changes
        '''
        if copy:
            frame = self.copy()
        else:
            frame = self

        # Check whether any observations are dropped
        prev_len = len(frame)

        if frame._col_included('t'):
            # Temporarily disable warnings
            warnings.filterwarnings('ignore')
            if how not in ['max', max]:
                # Group by worker id, time, and firm id, and take `how` of compensation
                frame.loc[:, 'y'] = frame.groupby(['i', 't', 'j'])['y'].transform(how)
            # Take max over duplicates
            # Source: https://stackoverflow.com/questions/23394476/keep-other-columns-when-doing-groupby
            frame = frame.loc[frame.loc[:, 'y'].to_numpy() == frame.groupby(['i', 't'])['y'].transform(max).to_numpy(), :].groupby(['i', 't'], as_index=False).first()
            # Restore warnings
            warnings.filterwarnings('default')

        # Data now has unique i-t observations
        frame.i_t_unique = True

        # If observations dropped, recompute 'm'
        if prev_len != len(frame):
            frame = frame.gen_m(force=True, copy=False)

        return frame

    def _conset(self, connectedness='connected', how_max='length', drop_multiples=False, copy=True):
        '''
        Update data to include only the largest connected component of movers.

        Arguments:
            connectedness (str or None): if 'connected', keep observations in the largest connected component of firms; if 'biconnected', keep observations in the largest biconnected component of firms; if None, keep all observations
            how_max (str): how to determine largest biconnected component. Options are 'length', 'firms', and 'workers', where each option chooses the biconnected component with the highest of the chosen value
            drop_multiples (bool): if True, rather than collapsing over spells, drop any spells with multiple observations (this is for computational efficiency when re-collapsing data for biconnected components)
            copy (bool): if False, avoid copy

        Returns:
            frame (BipartiteBase): BipartiteBase with (bi)connected component of movers
        '''
        if copy:
            frame = self.copy()
        else:
            frame = self

        if connectedness is None:
            # Skipping connected set
            frame.connected = None
            # frame._check_contiguous_ids() # This is necessary
            return frame

        # Keep track of whether contiguous ids change
        prev_workers = frame.n_workers()
        prev_firms = frame.n_firms()
        prev_clusters = frame.n_clusters()

        # Update data
        # Find largest connected set of firms
        # First, create graph
        G = frame._construct_graph(connectedness)
        if connectedness == 'connected':
            # Compute largest connected set of firms
            largest_cc = max(G.components(), key=len)
            # Keep largest connected set of firms
            frame = frame.keep_ids('j', largest_cc, copy=False)
        elif connectedness == 'biconnected_observations':
            if isinstance(frame, bpd.BipartiteEventStudyBase):
                # warnings.warn('You should avoid computing biconnected components on event study data. This is because intermediate firms can be dropped, for instance a worker may work at firms A, B, then return to A. But if B is not in the largest biconnected set of firms, the worker now has their associated firms listed as A, A. This causes two issues. First, for event study format, this could lead to situations that require complicated data rearrangement. For instance, the second observation in the first row might have to be dropped. This would require shifting the first observation of the second row into the second observation of the first row, etc. Second, this could turn a mover into a stayer. This is an issue for collapsed event study data - these observations should be collapsed into a single observation. Neither of these issues is handled properly in the current implementation - both are handled by simply dropping rows where either firm is outside the biconnected components. Please compute biconnected components on data in long or collapsed long format, then convert to event study format.')
                warnings.warn('You should avoid computing biconnected components on event study data. It requires converting data into long format and back into event study format, which is computationally expensive.')
            # Compute all connected components of firms (each entry is a connected component)
            cc_list = G.components()
            # Keep largest leave-one-out set of firms
            frame = frame._leave_one_observation_out(cc_list=cc_list, how_max=how_max, drop_multiples=drop_multiples)
        elif connectedness == 'biconnected_firms':
            if isinstance(frame, bpd.BipartiteEventStudyBase):
                warnings.warn('You should avoid computing biconnected components on event study data. It requires converting data into long format and back into event study format, which is computationally expensive.')
            # Compute all biconnected components of firms (each entry is a biconnected component)
            bcc_list = G.biconnected_components()
            # Keep largest leave-one-out set of firms
            frame = frame._leave_one_firm_out(bcc_list=bcc_list, how_max=how_max, drop_multiples=drop_multiples)
        else:
            raise NotImplementedError("Invalid connectedness: {}. Valid options are 'connected', 'biconnected', and None.".format(connectedness))

        # Data is now connected
        frame.connected = connectedness

        # If connected data != full data, set contiguous to False
        if prev_workers != frame.n_workers():
            frame.columns_contig['i'] = False
        if prev_firms != frame.n_firms():
            frame.columns_contig['j'] = False
        if prev_clusters is not None and prev_clusters != frame.n_clusters():
            frame.columns_contig['g'] = False

        return frame

    def _construct_graph(self, connectedness='connected'):
        '''
        Construct igraph graph linking firms by movers.

        Arguments:
            connectedness (str): if 'connected', for use when computing the largest connected component of firms; if 'biconnected_observations' or 'biconnected_firms', for use when computing the largest leave-one-out component of firms

        Returns:
            (igraph Graph): graph
        '''
        linkages_fn_dict = {
            'connected': self._construct_connected_linkages,
            'biconnected_observations': self._construct_biconnected_linkages,
            'biconnected_firms': self._construct_biconnected_linkages
        }
        # n_firms = self.loc[(self.loc[:, 'm'] > 0).to_numpy(), :].n_firms()
        return ig.Graph(edges=linkages_fn_dict[connectedness]()) # n=n_firms

    def sort_cols(self, copy=True):
        '''
        Sort frame columns (not in-place).

        Arguments:
            copy (bool): if False, avoid copy

        Returns:
            frame (BipartiteBase): BipartiteBase with columns sorted
        '''
        if copy:
            frame = self.copy()
        else:
            frame = self

        # Sort columns
        sorted_cols = sorted(frame.columns, key=col_order)
        frame = frame.reindex(sorted_cols, axis=1, copy=False)

        return frame

    def drop_rows(self, rows, drop_multiples=False, reset_index=True, copy=True):
        '''
        Drop particular rows.

        Arguments:
            rows (list): rows to keep
            drop_multiples (bool): used only if using collapsed format. If True, rather than collapsing over spells, drop any spells with multiple observations (this is for computational efficiency)
            reset_index (bool): if True, reset index at end
            copy (bool): if False, avoid copy

        Returns:
            frame (BipartiteBase): dataframe with given rows dropped
        '''
        rows = set(rows)
        if len(rows) == 0:
            # If nothing input
            if copy:
                return self.copy()
            return self
        self_rows = set(self.index.to_numpy())
        rows_diff = self_rows.difference(rows)

        return self.keep_rows(rows_diff, drop_multiples=drop_multiples, reset_index=reset_index, copy=copy)

    def min_movers_firms(self, threshold=15):
        '''
        List firms with at least `threshold` many movers.

        Arguments:
            threshold (int): minimum number of movers required to keep a firm

        Returns:
            valid_firms (NumPy Array): firms with sufficiently many movers
        '''
        if threshold == 0:
            # If no threshold
            return self.unique_ids('j')

        frame = self.loc[self.loc[:, 'm'].to_numpy() > 0, :]

        return frame.min_workers_firms(threshold)

    @recollapse_loop(True)
    def min_movers_frame(self, threshold=15, drop_multiples=False, copy=True):
        '''
        Return dataframe of firms with at least `threshold` many movers.

        Arguments:
            threshold (int): minimum number of movers required to keep a firm
            drop_multiples (bool): used only for collapsed format. If True, rather than collapsing over spells, drop any spells with multiple observations (this is for computational efficiency)
            copy (bool): if False, avoid copy

        Returns:
            (BipartiteBase): dataframe of firms with sufficiently many movers
        '''
        if threshold == 0:
            # If no threshold
            if copy:
                return self.copy()
            return self

        valid_firms = self.min_movers_firms(threshold)

        return self.keep_ids('j', keep_ids=valid_firms, drop_multiples=drop_multiples, copy=copy)

    def _prep_cluster_data(self, stayers_movers=None, t=None, weighted=True):
        '''
        Prepare data for clustering.

        Arguments:
            stayers_movers (str or None, default=None): if None, clusters on entire dataset; if 'stayers', clusters on only stayers; if 'movers', clusters on only movers
            t (int or None, default=None): if None, clusters on entire dataset; if int, gives period in data to consider (only valid for non-collapsed data)
            weighted (bool, default=True): if True, weight firm clusters by firm size (if a weight column is included, firm weight is computed using this column; otherwise, each observation has weight 1)
        Returns:
            data (Pandas DataFrame): data prepared for clustering
            weights (NumPy Array or None): if weighted=True, gives NumPy array of firm weights for clustering; otherwise, is None
            jids (NumPy Array): firm ids of firms in subset of data used to cluster
        '''
        # Prepare data
        # Stack data if event study (need all data in 1 column)
        if isinstance(self, bpd.BipartiteEventStudyBase):
            # Returns Pandas dataframe, not BipartiteLong(Collapsed)
            frame = self.get_long(return_df=True)
        else:
            frame = self

        if stayers_movers is not None:
            if stayers_movers == 'stayers':
                data = pd.DataFrame(frame.loc[frame.loc[:, 'm'].to_numpy() == 0, :])
            elif stayers_movers == 'movers':
                data = pd.DataFrame(frame.loc[frame.loc[:, 'm'].to_numpy() > 0, :])
        else:
            data = pd.DataFrame(frame)

        # If period-level, then only use data for that particular period
        if t is not None:
            if len(to_list(self.reference_dict['t'])) == 1:
                data = data.loc[data.loc[:, 't'].to_numpy() == t, :]
            else:
                warnings.warn('Cannot use data from a particular period on collapsed data, proceeding to cluster on all data')

        # Create weights
        if weighted:
            if self._col_included('w'):
                data.loc[:, 'row_weights'] = data.loc[:, 'w']
                weights = data.groupby('j')['w'].sum().to_numpy()
            else:
                data.loc[:, 'row_weights'] = 1
                weights = data.groupby('j').size().to_numpy()
        else:
            data.loc[:, 'row_weights'] = 1
            weights = None

        # Get unique firm ids
        jids = sorted(data.loc[:, 'j'].unique()) # Must sort

        return data, weights, jids

    def cluster(self, measures=bpd.measures.cdfs(), grouping=bpd.grouping.kmeans(), stayers_movers=None, t=None, weighted=True, dropna=False, copy=False):
        '''
        Cluster data and assign a new column giving the cluster for each firm.

        Arguments:
            measures (function or list of functions): how to compute measures for clustering. Options can be seen in bipartitepandas.measures.
            grouping (function): how to group firms based on measures. Options can be seen in bipartitepandas.grouping.
            stayers_movers (str or None): if None, clusters on entire dataset; if 'stayers', clusters on only stayers; if 'movers', clusters on only movers
            t (int or None): if None, clusters on entire dataset; if int, gives period in data to consider (only valid for non-collapsed data)
            weighted (bool): if True, weight firm clusters by firm size (if a weight column is included, firm weight is computed using this column; otherwise, each observation has weight 1)
            dropna (bool): if True, drop observations where firms aren't clustered; if False, keep all observations
            copy (bool): if False, avoid copy

        Returns:
            frame (BipartiteBase): BipartiteBase with clusters
        '''
        if copy:
            frame = self.copy()
        else:
            frame = self

        # Prepare data for clustering
        cluster_data, weights, jids = self._prep_cluster_data(stayers_movers=stayers_movers, t=t, weighted=weighted)

        # Compute measures
        for i, measure in enumerate(to_list(measures)):
            if i == 0:
                computed_measures = measure(cluster_data, jids)
            else:
                # For computing both cdfs and moments
                computed_measures = np.concatenate([computed_measures, measure(cluster_data, jids)], axis=1)
        frame.logger.info('firm moments computed')

        # Can't group using quantiles if more than 1 column
        if (grouping.__name__ == 'compute_quantiles') and (computed_measures.shape[1] > 1):
            grouping = bpd.measures.kmeans()
            warnings.warn('Cannot cluster using quantiles if multiple measures computed. Defaulting to KMeans.')

        # Compute firm groups
        frame.logger.info('computing firm groups')
        clusters = grouping(computed_measures, weights)
        frame.logger.info('firm groups computed')

        # Drop columns (because prepared data is not always a copy, must drop from self)
        for col in ['row_weights', 'one']:
            if col in self.columns:
                self.drop(col)

        # Drop existing clusters
        if frame._col_included('g'):
            frame.drop('g')

        for i, j_col in enumerate(to_list(frame.reference_dict['j'])):
            if len(to_list(self.reference_dict['j'])) == 1:
                g_col = 'g'
            elif len(to_list(self.reference_dict['j'])) == 2:
                g_col = 'g' + str(i + 1)
            clusters_dict = {j_col: jids, g_col: clusters}
            clusters_df = pd.DataFrame(clusters_dict, index=np.arange(len(jids)))
            frame.logger.info('dataframe linking fids to clusters generated')

            # Merge into event study data
            frame = frame.merge(clusters_df, how='left', on=j_col, copy=False)
            # Keep column as int even with nans
            frame.loc[:, g_col] = frame.loc[:, g_col].astype('Int64', copy=False)
            frame.col_dict[g_col] = g_col

        # Sort columns
        frame = frame.sort_cols(copy=False)

        if dropna:
            # Drop firms that don't get clustered
            frame.dropna(inplace=True)
            frame.reset_index(drop=True, inplace=True)
            frame.loc[:, frame.reference_dict['g']] = frame.loc[:, frame.reference_dict['g']].astype(int, copy=False)
            frame.clean_data({'connectedness': frame.connected}) # Compute same connectedness measure

        frame.columns_contig['g'] = True

        frame.logger.info('clusters merged into data')

        return frame
