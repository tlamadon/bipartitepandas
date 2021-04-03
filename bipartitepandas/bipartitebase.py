'''
Class for a bipartite network
'''
from pandas.core.indexes.base import InvalidIndexError
from tqdm.auto import tqdm
import numpy as np
# from numpy_groupies.aggregate_numpy import aggregate
import pandas as pd
from pandas import DataFrame, Int64Dtype
import networkx as nx
from sklearn.cluster import KMeans
from scipy.sparse.csgraph import connected_components
import warnings
from bipartitepandas import col_order, update_dict, to_list, logger_init, col_dict_optional_cols, aggregate_transform

class BipartiteBase(DataFrame):
    '''
    Base class for BipartitePandas, where BipartitePandas gives a bipartite network of firms and workers. Contains generalized methods. Inherits from DataFrame.

    Arguments:
        *args: arguments for Pandas DataFrame
        columns_req (list): required columns (only put general column names for joint columns, e.g. put 'fid' instead of 'f1i', 'f2i'; then put the joint columns in reference_dict)
        columns_opt (list): optional columns (only put general column names for joint columns, e.g. put 'g' instead of 'g1', 'g2'; then put the joint columns in reference_dict)
        columns_contig (dictionary): columns requiring contiguous ids linked to boolean of whether those ids are contiguous, or None if column(s) not included, e.g. {'i': False, 'j': False, 'g': None} (only put general column names for joint columns)
        reference_dict (dict): clarify which columns are associated with a general column name, e.g. {'i': 'i', 'j': ['j1', 'j2']}
        col_dtype_dict (dict): link column to datatype
        col_dict (dict or None): make data columns readable. Keep None if column names already correct
        include_id_reference_dict (bool): if True, create dictionary of Pandas dataframes linking original id values to contiguous id values
        **kwargs: keyword arguments for Pandas DataFrame
    '''
    _metadata = ['col_dict', 'reference_dict', 'id_reference_dict', 'col_dtype_dict', 'columns_req', 'columns_opt', 'columns_contig', 'default_KMeans', 'default_cluster', 'dtype_dict', 'connected', 'correct_cols', 'no_na', 'no_duplicates', 'i_t_unique'] # Attributes, required for Pandas inheritance

    def __init__(self, *args, columns_req=[], columns_opt=[], columns_contig=[], reference_dict={}, col_dtype_dict={}, col_dict=None, include_id_reference_dict=False, **kwargs):
        # Initialize DataFrame
        super().__init__(*args, **kwargs)

        # Start logger
        logger_init(self)
        # self.logger.info('initializing BipartiteBase object')

        if len(args) > 0 and isinstance(args[0], BipartiteBase): # Note that isinstance works for subclasses
            self.set_attributes(args[0])
        else:
            self.columns_req = ['i', 'j', 'y'] + columns_req
            self.columns_opt = ['g', 'm'] + columns_opt
            self.columns_contig = update_dict({'i': False, 'j': False, 'g': None}, columns_contig)
            self.reference_dict = update_dict({'i': 'i', 'm': 'm'}, reference_dict)
            if include_id_reference_dict:
                self.id_reference_dict = {id_col: pd.DataFrame() for id_col in self.reference_dict.keys()} # Link original id values to contiguous id values
            else:
                self.id_reference_dict = {}
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
            self.reset_attributes()

        # Define default parameter dictionaries
        self.default_KMeans = {
            'n_clusters': 10,
            'init': 'k-means++',
            'n_init': 500,
            'max_iter': 300,
            'tol': 0.0001,
            'precompute_distances': 'deprecated',
            'verbose': 0,
            'random_state': None,
            'copy_x': True,
            'n_jobs': 'deprecated',
            'algorithm': 'auto'
        }

        self.default_cluster = {
            'cdf_resolution': 10,
            'grouping': 'quantile_all',
            'stayers_movers': None,
            't': None,
            'dropna': False,
            'weighted': True,
            'user_KMeans': self.default_KMeans
        }

        self.dtype_dict = {
            'int': ['int', 'int8', 'int16', 'int32', 'int64', 'Int64'],
            'float': ['float', 'float8', 'float16', 'float32', 'float64', 'float128', 'int', 'int8', 'int16', 'int32', 'int64', 'Int64'],
            'str': 'str'
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
        bdf_copy.set_attributes(self) # This copies attribute dictionaries, default copy does not

        return bdf_copy

    def summary(self):
        '''
        Print summary statistics.
        '''
        mean_wage = np.mean(self[self.reference_dict['y']])
        max_wage = np.max(self[self.reference_dict['y']])
        min_wage = np.min(self[self.reference_dict['y']])
        ret_str = 'format: ' + type(self).__name__ + '\n'
        ret_str += 'number of workers: ' + str(self.n_workers()) + '\n'
        ret_str += 'number of firms: ' + str(self.n_firms()) + '\n'
        ret_str += 'number of observations: ' + str(len(self)) + '\n'
        ret_str += 'mean wage: ' + str(mean_wage) + '\n'
        ret_str += 'max wage: ' + str(max_wage) + '\n'
        ret_str += 'min wage: ' + str(min_wage) + '\n'
        ret_str += 'connected: ' + str(self.connected) + '\n'
        for contig_col, is_contig in self.columns_contig.items():
            ret_str += 'contiguous {} ids (None if not included): '.format(contig_col) + str(is_contig) + '\n'
        ret_str += 'correct column names and types: ' + str(self.correct_cols) + '\n'
        ret_str += 'no nans: ' + str(self.no_na) + '\n'
        ret_str += 'no duplicates: ' + str(self.no_duplicates) + '\n'
        ret_str += 'i-t (worker-year) observations unique (None if t column(s) not included): ' + str(self.i_t_unique) + '\n'

        print(ret_str)

    def n_workers(self):
        '''
        Get the number of unique workers.

        Returns:
            (int): number of unique workers
        '''
        return len(self['i'].unique())

    def n_firms(self):
        '''
        Get the number of unique firms.

        Returns:
            (int): number of unique firms
        '''
        fid_lst = []
        for fid_col in to_list(self.reference_dict['j']):
            fid_lst += list(self[fid_col].unique())
        return len(set(fid_lst))

    def n_clusters(self):
        '''
        Get the number of unique clusters.

        Returns:
            (int or None): number of unique clusters, None if not clustered
        '''
        if not self.col_included('g'): # If cluster column not in dataframe
            return None
        cid_lst = []
        for g_col in to_list(self.reference_dict['g']):
            cid_lst += list(self[g_col].unique())
        return len(set(cid_lst))

    def set_attributes(self, frame, no_dict=False):
        '''
        Set class attributes to equal those of another BipartitePandas object.

        Arguments:
            frame (BipartitePandas): BipartitePandas object whose attributes to use
            no_dict (bool): if True, only set booleans, no dictionaries
        '''
        # Dictionaries
        if not no_dict:
            self.columns_req = frame.columns_req.copy()
            self.columns_opt = frame.columns_opt.copy()
            self.reference_dict = frame.reference_dict.copy()
            self.col_dtype_dict = frame.col_dtype_dict.copy()
            self.col_dict = frame.col_dict.copy()
        self.columns_contig = frame.columns_contig.copy() # Required, even if no_dict
        self.id_reference_dict = {} # Required, even if no_dict
        if frame.id_reference_dict:
            # Must do a deep copy
            for id_col, reference_df in frame.id_reference_dict.items():
                self.id_reference_dict[id_col] = reference_df.copy()
        # Booleans
        self.connected = frame.connected # If True, all firms are connected by movers
        self.correct_cols = frame.correct_cols # If True, column names are correct
        self.no_na = frame.no_na # If True, no NaN observations in the data
        self.no_duplicates = frame.no_duplicates # If True, no duplicate rows in the data
        self.i_t_unique = frame.i_t_unique # If True, each worker has at most one observation per period

    def reset_attributes(self):
        '''
        Reset class attributes conditions to be False/None.
        '''
        for contig_col in self.columns_contig.keys():
            if self.col_included(contig_col):
                self.columns_contig[contig_col] = False
            else:
                self.columns_contig[contig_col] = None
        self.connected = False # If True, all firms are connected by movers
        self.correct_cols = False # If True, column names are correct
        self.no_na = False # If True, no NaN observations in the data
        self.no_duplicates = False # If True, no duplicate rows in the data
        self.i_t_unique = None # If True, each worker has at most one observation per period; if None, t column not included (set to False later in method if t column included)

        # Verify whether period included
        if self.col_included('t'):
            self.i_t_unique = False

    def col_included(self, col):
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

    def included_cols(self, flat=False):
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

    def drop(self, indices, axis=1, inplace=True):
        '''
        Drop indices along axis.

        Arguments:
            indices (int or str, optionally as a list): row(s) or column(s) to drop. For columns, use general column names for joint columns, e.g. put 'g' instead of 'g1', 'g2'. Only optional columns may be dropped
            axis (int): 0 to drop rows, 1 to drop columns
            inplace (bool): if True, modify in-place

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
                    elif col not in frame.included_cols() and col not in frame.included_cols(flat=True): # If column is not pre-established
                        DataFrame.drop(frame, col, axis=1, inplace=True)
                    else:
                        warnings.warn("{} is either (a) a required column and cannot be dropped or (b) a subcolumn that can be dropped, but only by specifying the general column name (e.g. use 'g' instead of 'g1' or 'g2')".format(col))
                else:
                    warnings.warn('{} is not in data columns'.format(col))
        elif axis == 0:
            DataFrame.drop(frame, indices, axis=0, inplace=True)
            frame.reset_attributes()
            frame.clean_data()

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
                elif col_cur not in frame.included_cols() and col_cur not in frame.included_cols(flat=True): # If column is not pre-established
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
            frame.set_attributes(self)
        return frame

    def contiguous_ids(self, id_col):
        '''
        Make column of ids contiguous.

        Arguments:
            id_col (str): column to make contiguous ('fid', 'wid', or 'j'). Use general column names for joint columns, e.g. put 'j' instead of 'j1', 'j2'. Only optional columns may be renamed

        Returns:
            frame (BipartiteBase): BipartiteBase with contiguous ids
        '''
        frame = self.copy()

        # Create sorted set of unique ids
        ids = []
        for id in to_list(self.reference_dict[id_col]):
            ids += list(frame[id].unique())
        ids = sorted(list(set(ids)))

        # Create list of adjusted ids
        adjusted_ids = np.arange(len(ids)).astype(int)

        # Save id reference dataframe, so user can revert back to original ids
        if frame.id_reference_dict: # If id_reference_dict has been initialized
            if len(frame.id_reference_dict[id_col]) == 0: # If dataframe empty, start with original ids: adjusted ids
                frame.id_reference_dict[id_col]['original_ids'] = ids
                frame.id_reference_dict[id_col]['adjusted_ids_1'] = adjusted_ids
            else: # Merge in new adjustment step
                n_cols = len(frame.id_reference_dict[id_col].columns)
                id_reference_df = pd.DataFrame({'adjusted_ids_' + str(n_cols - 1): ids, 'adjusted_ids_' + str(n_cols): adjusted_ids}, index=adjusted_ids).astype('Int64')
                frame.id_reference_dict[id_col] = frame.id_reference_dict[id_col].merge(id_reference_df, how='left', on='adjusted_ids_' + str(n_cols - 1))

        # Update each fid one at a time
        for id in to_list(self.reference_dict[id_col]):
            # Create dictionary linking current to new ids, then convert into a dataframe for merging
            ids_dict = {id: ids, 'adj_' + id: adjusted_ids}
            ids_df = pd.DataFrame(ids_dict, index=adjusted_ids)

            # Merge new, contiguous ids into event study data
            frame = frame.merge(ids_df, how='left', on=id)

            # Adjust id column to use new contiguous id
            frame[id] = frame['adj_' + id]
            frame.drop('adj_' + id)

        # Sort columns
        sorted_cols = sorted(frame.columns, key=col_order)
        frame = frame[sorted_cols]

        # ids are now contiguous
        frame.columns_contig[id_col] = True

        return frame

    def update_cols(self, inplace=True):
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

    def clean_data(self):
        '''
        Clean data to make sure there are no NaN or duplicate observations, firms are connected by movers and firm ids are contiguous.

        Returns:
            frame (BipartiteBase): BipartiteBase with cleaned data
        '''
        frame = self.copy()

        frame.logger.info('beginning BipartiteBase data cleaning')

        # First, correct columns
        # Note this must be done before data_validity(), otherwise certain checks are not guaranteed to work
        frame.logger.info('correcting columns')
        frame.update_cols()

        frame.logger.info('checking quality of data')
        # Make sure data is valid - computes correct_cols, no_na, no_duplicates, connected, and contiguous, along with other checks (note that column names are corrected in data_validity() if all columns are in the data)
        frame = BipartiteBase.data_validity(frame) # Shared data_validity

        # Next, drop NaN observations
        if not frame.no_na:
            frame.logger.info('dropping NaN observations')
            frame.dropna(inplace=True)

            # Update no_na
            frame.no_na = True

        # Next, drop duplicate observations
        if not frame.no_duplicates:
            frame.logger.info('dropping duplicate observations')
            frame.drop_duplicates(inplace=True)

            # Update no_duplicates
            frame.no_duplicates = True

        # Next, make sure i-t (worker-year) observations are unique
        if frame.i_t_unique is not None and not frame.i_t_unique:
            frame.logger.info('keeping highest paying job for i-t (worker-year) duplicates')
            frame.drop_i_t_duplicates()

        # Next, find largest set of firms connected by movers
        if not frame.connected:
            # Generate largest connected set
            frame.logger.info('generating largest connected set')
            frame = frame.conset()

        # Next, check contiguous ids
        for contig_col, is_contig in frame.columns_contig.items():
            if is_contig is not None and not is_contig:
                frame.logger.info('making {} ids contiguous'.format(contig_col))
                frame = frame.contiguous_ids(contig_col)

        # Using contiguous fids, get NetworkX Graph of largest connected set (note that this must be done even if firms already connected and contiguous)
        # frame.logger.info('generating NetworkX Graph of largest connected set')
        # _, frame.G = frame.conset(return_G=True) # FIXME currently not used

        # Sort columns
        frame.logger.info('sorting columns')
        sorted_cols = sorted(frame.columns, key=col_order)
        frame = frame[sorted_cols]

        frame.logger.info('BipartiteBase data cleaning complete')

        return frame

    def data_validity(self):
        '''
        Checks that data is formatted correctly and updates relevant attributes.

        Returns:
            frame (BipartiteBase): BipartiteBase with corrected attributes
        '''
        frame = self # .copy()

        success = True

        frame.logger.info('--- checking columns ---')
        all_cols = frame.included_cols()
        cols = True
        frame.logger.info('--- checking column datatypes ---')
        col_dtypes = True
        for col in all_cols:
            for subcol in to_list(frame.reference_dict[col]):
                if frame.col_dict[subcol] not in frame.columns:
                    frame.logger.info('{} missing from data'.format(frame.col_dict[subcol]))
                    col_dtypes = False
                    cols = False
                else:
                    col_type = str(frame[frame.col_dict[subcol]].dtype)
                    valid_types = to_list(frame.dtype_dict[frame.col_dtype_dict[col]])
                    if col_type not in valid_types:
                        frame.logger.info('{} has wrong dtype, should be {} but is {}'.format(frame.col_dict[subcol], frame.col_dtype_dict[col], col_type))
                        if col in frame.columns_contig.keys(): # If column contiguous
                            frame.logger.info('{} has wrong dtype, converting to contiguous integers'.format(frame.col_dict[subcol]))
                            frame = frame.contiguous_ids(col)
                        else:
                            frame.logger.info('{} has wrong dtype. Please check that this is the correct column (it is supposed to give {}), and if it is, cast it to a valid datatype (these include: {})'.format(frame.col_dict[subcol], subcol, valid_types))
                            col_dtypes = False
                            cols = False

        frame.logger.info('column datatypes correct:' + str(col_dtypes))
        if not col_dtypes:
            success = False
            raise ValueError('Your data does not include the correct columns or column datatypes. The BipartitePandas object cannot be generated with your data.')

        frame.logger.info('--- checking column names ---')
        col_names = True
        for col in all_cols:
            for subcol in to_list(frame.reference_dict[col]):
                if frame.col_dict[subcol] != subcol:
                    col_names = False
                    cols = False
                    break

        frame.logger.info('column names correct:' + str(col_names))
        if not col_names:
            success = False

        frame.logger.info('--- checking non-pre-established columns ---')
        all_cols = frame.included_cols(flat=True)
        for col in frame.columns:
            if col not in all_cols:
                cols = False
                break

        frame.logger.info('columns correct:' + str(cols))
        if not cols:
            frame.correct_cols = False
        else:
            frame.correct_cols = True

        frame.logger.info('--- checking nan data ---')
        nans = frame.shape[0] - frame.dropna().shape[0]

        frame.logger.info('data nans (should be 0):' + str(nans))
        if nans > 0:
            frame.no_na = False
            success = False
        else:
            frame.no_na = True

        frame.logger.info('--- checking duplicates ---')
        duplicates = frame.shape[0] - frame.drop_duplicates().shape[0]

        frame.logger.info('duplicates (should be 0):' + str(duplicates))
        if duplicates > 0:
            frame.no_duplicates = False
            success = False
        else:
            frame.no_duplicates = True

        if frame.col_included('t'):
            frame.logger.info('--- checking i-t (worker-year) observations ---')
            max_obs = 1
            for t_col in to_list(frame.reference_dict['t']):
                max_obs_col = frame.groupby(['i', t_col]).size().max()
                max_obs = max(max_obs_col, max_obs)

            frame.logger.info('max number of i-t (worker-year) observations (should be 1):' + str(max_obs))
            if max_obs > 1:
                frame.i_t_unique = False
                success = False
            else:
                frame.i_t_unique = True
        else:
            frame.i_t_unique = None

        frame.logger.info('--- checking connected set ---')
        if len(to_list(frame.reference_dict['j'])) == 1:
            frame['j_max'] = frame.groupby(['i'])['j'].transform(max)
            G = nx.from_pandas_edgelist(frame, 'j', 'j_max')
            # Drop fid_max
            frame.drop('j_max', axis=1)
            largest_cc = max(nx.connected_components(G), key=len)
            outside_cc = frame[(~frame['j'].isin(largest_cc))].shape[0]
        elif len(to_list(frame.reference_dict['j'])) == 2:
            G = nx.from_pandas_edgelist(frame, 'j1', 'j2')
            largest_cc = max(nx.connected_components(G), key=len)
            outside_cc = frame[(~frame['j1'].isin(largest_cc)) | (~frame['j2'].isin(largest_cc))].shape[0]
        else:
            raise InvalidIndexError("Trying to create network with 3 or more edges is not possible. Please check df.reference_dict['j']")

        frame.logger.info('observations outside connected set (should be 0):' + str(outside_cc))
        if outside_cc > 0:
            frame.connected = False
            success = False
        else:
            frame.connected = True

        # Check contiguous columns
        for contig_col, is_contig in frame.columns_contig.items():
            if frame.col_included(contig_col):
                frame.logger.info('--- checking contiguous {} ids ---'.format(contig_col))
                id_max = - np.inf
                ids_unique = []
                for id_col in to_list(frame.reference_dict[contig_col]):
                    id_max = max(frame[id_col].max(), id_max)
                    ids_unique = list(frame[id_col].unique()) + ids_unique
                n_ids = len(set(ids_unique))

                contig_ids = (id_max == n_ids - 1)
                frame.columns_contig[contig_col] = contig_ids

                frame.logger.info('contiguous {} ids (should be True): {}'.format(contig_col, contig_ids))
                if not contig_ids:
                    success = False
            else:
                frame.logger.info('--- skipping contiguous {} ids, column(s) not included ---'.format(contig_col))

        frame.logger.info('BipartiteBase success:' + str(success))

        return frame

    def drop_i_t_duplicates(self, inplace=True):
        '''
        Keep only the highest paying job for i-t (worker-year) duplicates.

        Arguments:
            inplace (bool): if True, modify in-place

        Returns:
            frame (BipartiteBase): BipartiteBase that keeps only the highest paying job for i-t (worker-year) duplicates. If no t column(s), returns frame with no changes
        '''
        if inplace:
            frame = self
        else:
            frame = self.copy()

        if frame.col_included('t'):
            t_cols = to_list(frame.reference_dict['t'])
            frame.sort_values(['i'] + t_cols + to_list(frame.reference_dict['y']), inplace=True)
            for t_col in t_cols:
                frame.drop_duplicates(subset=['i', t_col], keep='last', inplace=True)
        frame = frame.reset_index(drop=True)
        frame.i_t_unique = True

        return frame

    def conset(self, return_G=False):
        '''
        Update data to include only the largest connected set of movers, and if firm ids are contiguous, also return the NetworkX Graph.

        Arguments:
            return_G (bool): if True, return a tuple of (frame, G)

        Returns:
            frame (BipartiteBase): BipartiteBase with connected set of movers
            ALTERNATIVELY
            (tuple):
                frame (BipartiteBase): BipartiteBase with connected set of movers
                G (NetworkX Graph): largest connected set of movers (only returns if firm ids are contiguous, otherwise returns None)
        '''
        frame = self.copy()

        prev_workers = frame.n_workers()
        prev_firms = frame.n_firms()
        prev_clusters = frame.n_clusters()
        if len(to_list(self.reference_dict['j'])) == 1:
            # Add max firm id per worker to serve as a central node for the worker
            # frame['fid_f1'] = frame.groupby('wid')['fid'].transform(lambda a: a.shift(-1)) # FIXME - this is directed but is much slower
            frame['j_max'] = frame.groupby(['i'])['j'].transform(max) # FIXME - this is undirected but is much faster

            # Find largest connected set
            # Source: https://networkx.github.io/documentation/stable/reference/algorithms/generated/networkx.algorithms.components.connected_components.html
            G = nx.from_pandas_edgelist(frame, 'j', 'j_max')
            # Drop fid_max
            frame.drop('j_max', axis=1)
        elif len(to_list(self.reference_dict['j'])) == 2:
            G = nx.from_pandas_edgelist(frame, 'j1', 'j2')
        else:
            warnings.warn("Trying to create network with 3 or more edges is not possible. Please check df.reference_dict['j']. Returning unaltered frame")
            return frame
        # Update data if not connected
        if not frame.connected:
            largest_cc = max(nx.connected_components(G), key=len)
            # Keep largest connected set of firms
            if len(to_list(self.reference_dict['j'])) == 1:
                frame = frame[frame['j'].isin(largest_cc)]
            else:
                frame = frame[(frame['j1'].isin(largest_cc)) & (frame['j2'].isin(largest_cc))]

        # Data is now connected
        frame.connected = True

        # If connected data != full data, set contiguous to False
        if prev_workers != frame.n_workers():
            frame.columns_contig['i'] = False
        if prev_firms != frame.n_firms():
            frame.columns_contig['j'] = False
        if prev_clusters is not None and prev_clusters != frame.n_clusters():
            frame.columns_contig['g'] = False

        if return_G:
            # Return G if all ids are contiguous (if they're not contiguous, they have to be updated first)
            if frame.columns_contig['i'] and frame.columns_contig['j'] and (frame.columns_contig['g'] is None or frame.columns_contig['g']):
                return frame, G
            return frame, None
        return frame

    def gen_m(self, inplace=True):
        '''
        Generate m column for data (m == 0 if stayer, m == 1 if mover).

        Arguments:
            inplace (bool): if True, modify in-place

        Returns:
            frame (BipartiteBase): BipartiteBase with m column
        '''
        if inplace:
            frame = self
        else:
            frame = self.copy()

        if not frame.col_included('m'):
            if len(to_list(frame.reference_dict['j'])) == 1:
                frame['m'] = (aggregate_transform(frame, col_groupby='i', col_grouped='j', func='n_unique', col_name='m') > 1).astype(int)
            elif len(to_list(frame.reference_dict['j'])) == 2:
                frame['m'] = (frame['j1'] != frame['j2']).astype(int)
            else:
                warnings.warn("Trying to compute whether an individual moved firms is not possible with 3 or more firms per observation. Please check df.reference_dict['j']. Returning unaltered frame")
                return frame
            frame.col_dict['m'] = 'm'
            # Sort columns
            sorted_cols = sorted(frame.columns, key=col_order)
            frame = frame[sorted_cols]
        else:
            self.logger.info('m column already included. Returning unaltered frame')

        return frame

    def approx_cdfs(self, cdf_resolution=10, grouping='quantile_all', stayers_movers=None, t=None, weighted=True):
        '''
        Generate cdfs of compensation for firms.

        Arguments:
            cdf_resolution (int): how many values to use to approximate the cdfs
            grouping (str): how to group the cdfs ('quantile_all' to get quantiles from entire set of data, then have firm-level values between 0 and 1; 'quantile_firm_small' to get quantiles at the firm-level and have values be compensations if small data; 'quantile_firm_large' to get quantiles at the firm-level and have values be compensations if large data, note that this is up to 50 times slower than 'quantile_firm_small' and should only be used if the dataset is too large to copy into a dictionary)
            stayers_movers (str or None): if None, uses entire dataset; if 'stayers', uses only stayers; if 'movers', uses only movers
            t (int or None): if None, clusters on entire dataset; if int, gives period in data to consider (only valid for non-collapsed data)
            weighted (bool): if True, compute firm weights

        Returns:
            cdf_df (NumPy Array): NumPy array of firm cdfs
            weights (NumPy Array or None): if weighted=True, gives NumPy array of firm weights for clustering; otherwise, is None
            jids (NumPy Array): firm ids of firms in subset of data used to cluster
        '''
        if stayers_movers is not None: # Note this condition is split into two sections, one prior to stacking event study data and one post
            # Generate m column (the function checks if it already exists)
            self.gen_m()

        # Stack data if event study (need all data in 1 column)
        if len(to_list(self.reference_dict['j'])) == 2:
            frame = self.get_long(return_df=True) # Returns Pandas dataframe, not BipartiteLong(Collapsed)
        else:
            frame = self

        if stayers_movers is not None:
            if stayers_movers == 'stayers':
                data = pd.DataFrame(frame[frame['m'] == 0])
            elif stayers_movers == 'movers':
                data = pd.DataFrame(frame[frame['m'] == 1])
        else:
            data = pd.DataFrame(frame)

        # If period-level, then only use data for that particular period
        if t is not None:
            if len(to_list(self.reference_dict['t'])) == 1:
                data = data[data['t'] == t]
            else:
                warnings.warn('Cannot use data from a particular period on collapsed data, proceeding to cluster on all data')

        # Create empty numpy array to fill with the cdfs
        jids = sorted(data['j'].unique()) # Must sort
        n_firms = len(jids) # Can't use self.n_firms() since data could be a subset of self.data
        cdfs = np.zeros([n_firms, cdf_resolution])

        # Create weights
        if weighted:
            if self.col_included('w'):
                weights = data.groupby('j')['w'].sum()
            else:
                weights = data.groupby('j').size()
        else:
            weights = None

        # Create quantiles of interest
        quantiles = np.linspace(1 / cdf_resolution, 1, cdf_resolution)

        if grouping == 'quantile_all':
            # Get quantiles from all data
            quantile_groups = data['y'].quantile(quantiles)

            # Generate firm-level cdfs
            data.sort_values('j', inplace=True) # Required for aggregate_transform
            for i, quant in enumerate(quantile_groups):
                data['quant'] = (data['y'] <= quant).astype(int)
                cdfs_col = aggregate_transform(data, col_groupby='j', col_grouped='quant', func='sum', merge=False) # aggregate(data['fid'], firm_quant, func='sum', fill_value=- 1)
                cdfs[:, i] = cdfs_col[cdfs_col >= 0]
            data.drop('quant', axis=1, inplace=True)
            del cdfs_col

            # Normalize by firm size (convert to cdf)
            jsize = data.groupby('j').size().to_numpy()
            cdfs = (cdfs.T / jsize.T).T

        elif grouping in ['quantile_firm_small', 'quantile_firm_large']:
            # Sort data by compensation (do this once now, so that don't need to do it again later) (also note it is faster to sort then manually compute quantiles than to use built-in quantile functions)
            data = data.sort_values(['y'])

            if grouping == 'quantile_firm_small':
                # Convert pandas dataframe into a dictionary to access data faster
                # Source for idea: https://stackoverflow.com/questions/57208997/looking-for-the-fastest-way-to-slice-a-row-in-a-huge-pandas-dataframe
                # Source for how to actually format data correctly: https://stackoverflow.com/questions/56064677/pandas-series-to-dict-with-repeated-indices-make-dict-with-list-values
                # data_dict = data['y'].groupby(level=0).agg(list).to_dict()
                data_dict = data.groupby('j')['y'].agg(list).to_dict()
                # data.sort_values(['j', 'y'], inplace=True) # Required for aggregate_transform
                # data_dict = pd.Series(aggregate_transform(data, col_groupby='j', col_grouped='y', func='array', merge=False), index=np.unique(data['j'])).to_dict()
                # with warnings.catch_warnings():
                #     warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
                #     data_dict = pd.Series(aggregate(data['j'], data['y'], func='array', fill_value=[]), index=np.unique(data['j'])).to_dict()

            # Generate the cdfs
            for i, jid in enumerate(jids):
                # Get the firm-level compensation data (don't need to sort because already sorted)
                if grouping == 'quantile_firm_small':
                    y = data_dict[jid]
                elif grouping == 'quantile_firm_large':
                    y = data.loc[data['j'] == jid, 'y'].to_numpy()
                # Generate the firm-level cdf
                # Note: update numpy array element by element
                # Source: https://stackoverflow.com/questions/30012362/faster-way-to-convert-list-of-objects-to-numpy-array/30012403
                for j in range(cdf_resolution):
                    index = max(len(y) * (j + 1) // cdf_resolution - 1, 0) # Don't want negative index
                    # Update cdfs with the firm-level cdf
                    cdfs[i, j] = y[index]

        return cdfs, weights, jids

    def cluster(self, user_cluster={}):
        '''
        Cluster data and assign a new column giving the cluster for each firm.

        Arguments:
            user_cluster (dict): dictionary of parameters for clustering

                Dictionary parameters:

                    cdf_resolution (int): how many values to use to approximate the cdfs

                    grouping (str): how to group the cdfs ('quantile_all' to get quantiles from entire set of data, then have firm-level values between 0 and 1; 'quantile_firm_small' to get quantiles at the firm-level and have values be compensations if small data; 'quantile_firm_large' to get quantiles at the firm-level and have values be compensations if large data, note that this is up to 50 times slower than 'quantile_firm_small' and should only be used if the dataset is too large to copy into a dictionary)

                    stayers_movers (str or None): if None, clusters on entire dataset; if 'stayers', clusters on only stayers; if 'movers', clusters on only movers

                    t (int or None): if None, clusters on entire dataset; if int, gives period in data to consider (only valid for non-collapsed data)

                    weighted (bool): if True, weight firm clusters by firm size (if a weight column is included, firm weight is computed using this column; otherwise, each observation has weight 1)

                    dropna (bool): if True, drop observations where firms aren't clustered; if False, keep all observations

                    user_KMeans (dict): parameters for KMeans estimation (for more information on what parameters can be used, visit https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)

        Returns:
            frame (BipartiteBase): BipartiteBase with clusters
        '''
        frame = self.copy()

        # Update dictionary
        cluster_params = update_dict(frame.default_cluster, user_cluster)

        # Unpack dictionary
        cdf_resolution = cluster_params['cdf_resolution']
        grouping = cluster_params['grouping']
        stayers_movers = cluster_params['stayers_movers']
        t = cluster_params['t']
        weighted = cluster_params['weighted']
        user_KMeans = cluster_params['user_KMeans']

        # Compute cdfs
        cdfs, weights, jids = frame.approx_cdfs(cdf_resolution=cdf_resolution, grouping=grouping, stayers_movers=stayers_movers, t=t, weighted=weighted)
        frame.logger.info('firm cdfs computed')

        # Compute firm clusters
        KMeans_params = update_dict(frame.default_KMeans, user_KMeans)
        frame.logger.info('computing firm clusters')
        clusters = KMeans(**KMeans_params).fit(cdfs, sample_weight=weights).labels_
        frame.logger.info('firm clusters computed')

        # Drop existing clusters
        if frame.col_included('g'):
            frame.drop('g')

        for i, j_col in enumerate(to_list(frame.reference_dict['j'])):
            if len(to_list(self.reference_dict['j'])) == 1:
                g_col = 'g'
            elif len(to_list(self.reference_dict['j'])) == 2:
                g_col = 'g' + str(i + 1)
            print('j_col:', j_col)
            print('g_col:', g_col)
            clusters_dict = {j_col: jids, g_col: clusters}
            clusters_df = pd.DataFrame(clusters_dict, index=np.arange(len(jids)))
            frame.logger.info('dataframe linking fids to clusters generated')

            # Merge into event study data
            frame = frame.merge(clusters_df, how='left', on=j_col)
            # Keep column as int even with nans
            frame[g_col] = frame[g_col].astype('Int64')
            frame.col_dict[g_col] = g_col

        # Sort columns
        sorted_cols = sorted(frame.columns, key=col_order)
        frame = frame[sorted_cols]

        if cluster_params['dropna']:
            # Drop firms that don't get clustered
            frame = frame.dropna().reset_index(drop=True)
            frame[frame.reference_dict['g']] = frame[frame.reference_dict['g']].astype(int)
            frame.clean_data()

        frame.columns_contig['g'] = True

        frame.logger.info('clusters merged into data')

        return frame
