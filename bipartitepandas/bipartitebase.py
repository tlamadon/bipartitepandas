'''
Class for a bipartite network
'''
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
        columns_opt (list): optional columns (only put general column names for joint columns, e.g. put 'j' instead of 'j1', 'j2'; then put the joint columns in reference_dict)
        reference_dict (dict): clarify which columns are associated with a general column name, e.g. {'wid': 'wid', 'j': ['j1', 'j2']}
        col_dtype_dict (dict): link column to datatype
        col_dict (dict or None): make data columns readable. Keep None if column names already correct
        **kwargs: keyword arguments for Pandas DataFrame
    '''
    _metadata = ['col_dict', 'reference_dict', 'col_dtype_dict', 'columns_req', 'columns_opt', 'default_KMeans', 'default_cluster', 'dtype_dict', 'connected', 'contiguous_fids', 'contiguous_wids', 'contiguous_cids', 'correct_cols', 'no_na', 'no_duplicates', 'worker_year_unique'] # Attributes, required for Pandas inheritance

    def __init__(self, *args, columns_req=[], columns_opt=[], reference_dict={}, col_dtype_dict={}, col_dict=None, **kwargs):
        # Initialize DataFrame
        super().__init__(*args, **kwargs)

        # Start logger
        logger_init(self)
        # self.logger.info('initializing BipartiteBase object')

        if len(args) > 0 and isinstance(args[0], BipartiteBase): # Note that isinstance works for subclasses
            self.set_attributes(args[0])
        else:
            self.columns_req = columns_req + ['wid', 'fid', 'comp']
            self.columns_opt = columns_opt + ['m', 'j']
            self.reference_dict = update_dict({'wid': 'wid', 'm': 'm'}, reference_dict)
            self.col_dtype_dict = update_dict({'wid': 'int', 'fid': 'int', 'comp': 'float', 'year': 'int', 'm': 'int', 'j': 'int'}, col_dtype_dict)
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
            'year': None,
            'dropna': False,
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

    def summary(self):
        '''
        Print summary statistics.
        '''
        mean_wage = np.mean(self[self.reference_dict['comp']])
        max_wage = np.max(self[self.reference_dict['comp']])
        min_wage = np.min(self[self.reference_dict['comp']])
        ret_str = 'format: ' + type(self).__name__ + '\n'
        ret_str += 'number of workers: ' + str(self.n_workers()) + '\n'
        ret_str += 'number of firms: ' + str(self.n_firms()) + '\n'
        ret_str += 'number of observations: ' + str(len(self)) + '\n'
        ret_str += 'mean wage: ' + str(mean_wage) + '\n'
        ret_str += 'max wage: ' + str(max_wage) + '\n'
        ret_str += 'min wage: ' + str(min_wage) + '\n'
        ret_str += 'connected: ' + str(self.connected) + '\n'
        ret_str += 'contiguous firm ids: ' + str(self.contiguous_fids) + '\n'
        ret_str += 'contiguous worker ids: ' + str(self.contiguous_wids) + '\n'
        ret_str += 'contiguous cluster ids (None if not clustered): ' + str(self.contiguous_cids) + '\n'
        ret_str += 'correct column names and types: ' + str(self.correct_cols) + '\n'
        ret_str += 'no nans: ' + str(self.no_na) + '\n'
        ret_str += 'no duplicates: ' + str(self.no_duplicates) + '\n'
        ret_str += 'worker-year observations unique (None if year column(s) not included): ' + str(self.worker_year_unique) + '\n'

        print(ret_str)

    def n_workers(self):
        '''
        Get the number of unique workers.

        Returns:
            (int): number of unique workers
        '''
        return len(self['wid'].unique())

    def n_firms(self):
        '''
        Get the number of unique firms.

        Returns:
            (int): number of unique firms
        '''
        fid_lst = []
        for fid_col in to_list(self.reference_dict['fid']):
            fid_lst += list(self[fid_col].unique())
        return len(set(fid_lst))

    def n_clusters(self):
        '''
        Get the number of unique clusters.

        Returns:
            (int or None): number of unique clusters, None if not clustered
        '''
        if not self.col_included('j'): # If cluster column not in dataframe
            return None
        cid_lst = []
        for j_col in to_list(self.reference_dict['j']):
            cid_lst += list(self[j_col].unique())
        return len(set(cid_lst))

    def set_attributes(self, frame, deep=False, no_dict=False):
        '''
        Set class attributes to equal those of another BipartitePandas object.

        Arguments:
            frame (BipartitePandas): BipartitePandas object whose attributes to use
            deep (bool): if True, also copy dictionaries
            no_dict (bool): if True, only set booleans, no dictionaries
        '''
        # Dictionaries
        if not no_dict:
            if deep:
                self.columns_req = frame.columns_req.copy()
                self.columns_opt = frame.columns_opt.copy()
                self.reference_dict = frame.reference_dict.copy()
                self.col_dtype_dict = frame.col_dtype_dict.copy()
                self.col_dict = frame.col_dict.copy()
            else:
                self.columns_req = frame.columns_req
                self.columns_opt = frame.columns_opt
                self.reference_dict = frame.reference_dict
                self.col_dtype_dict = frame.col_dtype_dict
                self.col_dict = frame.col_dict
        # Booleans
        self.connected = frame.connected # If True, all firms are connected by movers
        self.contiguous_fids = frame.contiguous_fids # If True, firm ids are contiguous
        self.contiguous_wids = frame.contiguous_wids # If True, worker ids are contiguous
        self.contiguous_cids = frame.contiguous_cids # If True, cluster ids are contiguous; if None, data not clustered (set to False later in __init__ if clusters included)
        self.correct_cols = frame.correct_cols # If True, column names are correct
        self.no_na = frame.no_na # If True, no NaN observations in the data
        self.no_duplicates = frame.no_duplicates # If True, no duplicate rows in the data
        self.worker_year_unique = frame.worker_year_unique # If True, each worker has at most one observation per year

    def reset_attributes(self):
        '''
        Reset class attributes conditions to be False/None.
        '''
        self.connected = False # If True, all firms are connected by movers
        self.contiguous_fids = False # If True, firm ids are contiguous
        self.contiguous_wids = False # If True, worker ids are contiguous
        self.contiguous_cids = None # If True, cluster ids are contiguous; if None, data not clustered (set to False later in __init__ if clusters included)
        self.correct_cols = False # If True, column names are correct
        self.no_na = False # If True, no NaN observations in the data
        self.no_duplicates = False # If True, no duplicate rows in the data
        self.worker_year_unique = None # If True, each worker has at most one observation per year

        # Verify whether clusters included
        if self.col_included('j'):
            self.contiguous_cids = False

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
            indices (int or str, optionally as a list): row(s) or column(s) to drop. For columns, use general column names for joint columns, e.g. put 'j' instead of 'j1', 'j2'. Only optional columns may be dropped
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
                        if col == 'j':
                            frame.contiguous_cids = None
                    elif col not in frame.included_cols() and col not in frame.included_cols(flat=True): # If column is not pre-established
                        DataFrame.drop(frame, col, axis=1, inplace=True)
                    else:
                        warnings.warn("{} is either (a) a required column and cannot be dropped or (b) a subcolumn that can be dropped, but only by specifying the general column name (e.g. use 'j' instead of 'j1' or 'j2')".format(col))
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
            rename_dict (dict): key is current column name, value is new column name. Use general column names for joint columns, e.g. put 'j' instead of 'j1', 'j2'. Only optional columns may be renamed
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
                    if col_cur == 'j':
                        frame.contiguous_cids = None
                elif col_cur not in frame.included_cols() and col_cur not in frame.included_cols(flat=True): # If column is not pre-established
                        DataFrame.rename(frame, {col_cur: col_new}, axis=1, inplace=True)
                else:
                    warnings.warn("{} is either (a) a required column and cannot be renamed or (b) a subcolumn that can be renamed, but only by specifying the general column name (e.g. use 'j' instead of 'j1' or 'j2')".format(col_cur))
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

        if id_col == 'fid':
            # Firm ids are now contiguous
            frame.contiguous_fids = True
        elif id_col == 'wid':
            # Worker ids are now contiguous
            frame.contiguous_wids = True
        elif id_col == 'j':
            # Cluster ids are now contiguous
            frame.contiguous_cids = True

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
        BipartiteBase.data_validity(frame) # Shared data_validity

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

        # Next, make sure worker-year observations are unique
        if frame.worker_year_unique is not None and not frame.worker_year_unique:
            frame.logger.info('keeping highest paying job for worker-year duplicates')
            frame.drop_worker_year_duplicates()

            # Update no_duplicates
            frame.worker_year_unique = True

        # Next, find largest set of firms connected by movers
        if not frame.connected:
            # Generate largest connected set
            frame.logger.info('generating largest connected set')
            frame = frame.conset()

        # Next, make firm ids contiguous
        if not frame.contiguous_fids:
            frame.logger.info('making firm ids contiguous')
            frame = frame.contiguous_ids('fid')

        # Next, make worker ids contiguous
        if not frame.contiguous_wids:
            frame.logger.info('making worker ids contiguous')
            frame = frame.contiguous_ids('wid')

        # Next, make cluster ids contiguous
        if frame.contiguous_cids is not None and not frame.contiguous_cids:
            frame.logger.info('making cluster ids contiguous')
            frame = frame.contiguous_ids('j')

        # Using contiguous fids, get NetworkX Graph of largest connected set (note that this must be done even if firms already connected and contiguous)
        # frame.logger.info('generating NetworkX Graph of largest connected set')
        # _, frame.G = frame.conset(return_G=True) # FIXME currently not used

        # Sort columns
        frame.logger.info('sorting columns')
        sorted_cols = sorted(frame.columns, key=col_order)
        frame = frame[sorted_cols]

        frame.logger.info('BipartiteBase data cleaning complete')

        return frame

    def data_validity(self, inplace=True):
        '''
        Checks that data is formatted correctly and updates relevant attributes.

        Arguments:
            inplace (bool): if True, modify in-place

        Returns:
            frame (BipartiteBase): BipartiteBase with corrected attributes
        '''
        if inplace:
            frame = self
        else:
            frame = self.copy()

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
                        if col in ['fid', 'wid', 'j']:
                            frame.logger.info('{} has wrong dtype, converting to contiguous integers'.format(frame.col_dict[subcol]))
                            frame.contiguous_ids(col)
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

        if frame.col_included('year'):
            frame.logger.info('--- checking worker-year observations ---')
            max_obs = 1
            for year_col in to_list(frame.reference_dict['year']):
                max_obs_col = frame.groupby(['wid', year_col]).size().max()
                max_obs = max(max_obs_col, max_obs)

            frame.logger.info('max number of worker-year observations (should be 1):' + str(max_obs))
            if max_obs > 1:
                frame.worker_year_unique = False
                success = False
            else:
                frame.worker_year_unique = True
        else:
            frame.worker_year_unique = None

        frame.logger.info('--- checking connected set ---')
        if frame.reference_dict['fid'] == 'fid':
            frame['fid_max'] = frame.groupby(['wid'])['fid'].transform(max)
            G = nx.from_pandas_edgelist(frame, 'fid', 'fid_max')
            # Drop fid_max
            frame.drop('fid_max', axis=1)
            largest_cc = max(nx.connected_components(G), key=len)
            outside_cc = frame[(~frame['fid'].isin(largest_cc))].shape[0]
        else:
            G = nx.from_pandas_edgelist(frame, 'f1i', 'f2i')
            largest_cc = max(nx.connected_components(G), key=len)
            outside_cc = frame[(~frame['f1i'].isin(largest_cc)) | (~frame['f2i'].isin(largest_cc))].shape[0]

        frame.logger.info('observations outside connected set (should be 0):' + str(outside_cc))
        if outside_cc > 0:
            frame.connected = False
            success = False
        else:
            frame.connected = True

        frame.logger.info('--- checking contiguous firm ids ---')
        fid_max = - np.inf
        for fid_col in to_list(frame.reference_dict['fid']):
            fid_max = max(frame[fid_col].max(), fid_max)
        n_firms = frame.n_firms()

        contig_fids = (fid_max == n_firms - 1)
        frame.contiguous_fids = contig_fids

        frame.logger.info('contiguous firm ids (should be True):' + str(contig_fids))
        if not contig_fids:
            success = False

        frame.logger.info('--- checking contiguous worker ids ---')
        wid_max = frame['wid'].max()
        n_workers = frame.n_workers()

        contig_wids = (wid_max == n_workers - 1)
        frame.contiguous_wids = contig_wids

        frame.logger.info('contiguous worker ids (should be True):' + str(contig_wids))
        if not contig_wids:
            success = False

        if frame.col_included('j'):
            frame.logger.info('--- checking contiguous cluster ids ---')
            cid_max = - np.inf
            for cid_col in to_list(frame.reference_dict['j']):
                cid_max = max(frame[cid_col].max(), cid_max)
            n_cids = frame.n_clusters()

            contig_cids = (cid_max == n_cids - 1)
            frame.contiguous_cids = contig_cids

            frame.logger.info('contiguous cluster ids (should be True):' + str(contig_cids))
            if not contig_cids:
                success = False

        frame.logger.info('BipartiteBase success:' + str(success))

        return frame

    def drop_worker_year_duplicates(self, inplace=True):
        '''
        Keep only the highest paying job for worker-year duplicates.

        Arguments:
            inplace (bool): if True, modify in-place

        Returns:
            frame (BipartiteBase): BipartiteBase that keeps only the highest paying job for worker-year duplicates. If no year columns, returns frame with no changes
        '''
        if inplace:
            frame = self
        else:
            frame = self.copy()

        if 'year' in frame.included_cols():
            year_cols = to_list(frame.reference_dict['year'])
            frame.sort_values(['wid'] + year_cols + to_list(frame.reference_dict['comp']), inplace=True)
            for year_col in year_cols:
                frame.drop_duplicates(subset=['wid', year_col], keep='last', inplace=True)

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
        if self.reference_dict['fid'] == 'fid':
            # Add max firm id per worker to serve as a central node for the worker
            # frame['fid_f1'] = frame.groupby('wid')['fid'].transform(lambda a: a.shift(-1)) # FIXME - this is directed but is much slower
            frame['fid_max'] = frame.groupby(['wid'])['fid'].transform(max) # FIXME - this is undirected but is much faster

            # Find largest connected set
            # Source: https://networkx.github.io/documentation/stable/reference/algorithms/generated/networkx.algorithms.components.connected_components.html
            G = nx.from_pandas_edgelist(frame, 'fid', 'fid_max')
            # Drop fid_max
            frame.drop('fid_max', axis=1)
        else:
            G = nx.from_pandas_edgelist(frame, 'f1i', 'f2i')
        # Update data if not connected
        if not frame.connected:
            largest_cc = max(nx.connected_components(G), key=len)
            # Keep largest connected set of firms
            if self.reference_dict['fid'] == 'fid':
                frame = frame[frame['fid'].isin(largest_cc)]
            else:
                frame = frame[(frame['f1i'].isin(largest_cc)) & (frame['f2i'].isin(largest_cc))]

        # Data is now connected
        frame.connected = True

        # If connected data != full data, set contiguous to False
        if prev_firms != frame.n_firms():
            frame.contiguous_fids = False
        if prev_workers != frame.n_workers():
            frame.contiguous_wids = False
        if prev_clusters is not None and prev_clusters != frame.n_clusters():
            frame.contiguous_cids = False

        if return_G:
            # Return G if all ids are contiguous (if they're not contiguous, they have to be updated first)
            if frame.contiguous_fids and frame.contiguous_wids and (frame.col_dict['j'] is None or frame.contiguous_cids):
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
            if frame.reference_dict['fid'] == 'fid':
                frame['m'] = (aggregate_transform(frame, col_groupby='wid', col_grouped='fid', func='n_unique', col_name='m') > 1).astype(int)
            else:
                frame['m'] = (frame['f1i'] != frame['f2i']).astype(int)
            frame.col_dict['m'] = 'm'
            # Sort columns
            sorted_cols = sorted(frame.columns, key=col_order)
            frame = frame[sorted_cols]

        return frame

    def approx_cdfs(self, cdf_resolution=10, grouping='quantile_all', stayers_movers=None, year=None):
        '''
        Generate cdfs of compensation for firms.

        Arguments:
            cdf_resolution (int): how many values to use to approximate the cdf
            grouping (str): how to group the cdfs ('quantile_all' to get quantiles from entire set of data, then have firm-level values between 0 and 1; 'quantile_firm_small' to get quantiles at the firm-level and have values be compensations if small data; 'quantile_firm_large' to get quantiles at the firm-level and have values be compensations if large data, note that this is up to 50 times slower than 'quantile_firm_small' and should only be used if the dataset is too large to copy into a dictionary)
            stayers_movers (str or None): if None, uses entire dataset; if 'stayers', uses only stayers; if 'movers', uses only movers
            year (int or None): if None, uses entire dataset; if int, gives year of data to consider

        Returns:
            cdf_df (NumPy Array): NumPy array of firm cdfs
            fids (NumPy Array): firm ids of firms in subset of data used to cluster
        '''
        if stayers_movers is not None:
            # Determine whether m column exists
            if not self.col_included('m'):
                self.gen_m()
            if stayers_movers == 'stayers':
                data = pd.DataFrame(self[self['m'] == 0])
            elif stayers_movers == 'movers':
                data = pd.DataFrame(self[self['m'] == 1])
        else:
            data = pd.DataFrame(self)

        # If year-level, then only use data for that particular year
        if year is not None:
            if self.reference_dict['year'] == 'year':
                data = data[data['year'] == year]
            else:
                warnings.warn('Cannot use data from a particular year on non-BipartiteLong data. Convert into BipartiteLong to cluster only on a particular year')

        # Create empty numpy array to fill with the cdfs
        if self.reference_dict['fid'] == ['f1i', 'f2i']: # If Event Study
            # n_firms = len(set(list(data['f1i'].unique()) + list(data['f2i'].unique()))) # Can't use self.n_firms() since data could be a subset of self.data
            data = data.rename({'f1i': 'fid', 'y1': 'comp'}, axis=1)
            data = pd.concat([data, data.rename({'f2i': 'fid', 'y2': 'comp', 'fid': 'f2i', 'comp': 'y2'}, axis=1).assign(f2i = - 1)], axis=0) # Include irrelevant columns and rename f1i to f2i to prevent nans, which convert columns from int into float # FIXME duplicating both movers and stayers, should probably only be duplicating movers
        fids = data['fid'].unique()
        n_firms = len(fids) # Can't use self.n_firms() since data could be a subset of self.data
        cdfs = np.zeros([n_firms, cdf_resolution])

        # Create quantiles of interest
        quantiles = np.linspace(1 / cdf_resolution, 1, cdf_resolution)

        if grouping == 'quantile_all':
            # Get quantiles from all data
            quantile_groups = data['comp'].quantile(quantiles)

            # Generate firm-level cdfs
            data.sort_values('fid', inplace=True) # Required for aggregate_transform
            for i, quant in enumerate(quantile_groups):
                data = data.assign(firm_quant=data['comp'] <= quant).astype(int)
                cdfs_col = aggregate_transform(data, col_groupby='fid', col_grouped='firm_quant', func='sum', merge=False) # aggregate(data['fid'], firm_quant, func='sum', fill_value=- 1)
                cdfs[:, i] = cdfs_col[cdfs_col >= 0]
            data.drop('firm_quant', axis=1, inplace=True)
            del cdfs_col

            # Normalize by firm size (convert to cdf)
            fsize = data.groupby('fid').size().to_numpy()
            cdfs = (cdfs.T / fsize.T).T

        elif grouping in ['quantile_firm_small', 'quantile_firm_large']:
            # Sort data by compensation (do this once now, so that don't need to do it again later) (also note it is faster to sort then manually compute quantiles than to use built-in quantile functions)
            data = data.sort_values(['comp'])

            if grouping == 'quantile_firm_small':
                # Convert pandas dataframe into a dictionary to access data faster
                # Source for idea: https://stackoverflow.com/questions/57208997/looking-for-the-fastest-way-to-slice-a-row-in-a-huge-pandas-dataframe
                # Source for how to actually format data correctly: https://stackoverflow.com/questions/56064677/pandas-series-to-dict-with-repeated-indices-make-dict-with-list-values
                # data_dict = data['comp'].groupby(level=0).agg(list).to_dict()
                data_dict = data.groupby('fid')['comp'].agg(list).to_dict()
                # data.sort_values(['fid', 'comp'], inplace=True) # Required for aggregate_transform
                # data_dict = pd.Series(aggregate_transform(data, col_groupby='fid', col_grouped='comp', func='array', merge=False), index=np.unique(data['fid'])).to_dict()
                # with warnings.catch_warnings():
                #     warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
                #     data_dict = pd.Series(aggregate(data['fid'], data['comp'], func='array', fill_value=[]), index=np.unique(data['fid'])).to_dict()

            # Generate the cdfs
            for fid in range(n_firms):
                # Get the firm-level compensation data (don't need to sort because already sorted)
                if grouping == 'quantile_firm_small':
                    comp = data_dict[fid]
                elif grouping == 'quantile_firm_large':
                    comp = data.loc[data['fid'] == fid, 'comp'].to_numpy()
                # Generate the firm-level cdf
                # Note: update numpy array element by element
                # Source: https://stackoverflow.com/questions/30012362/faster-way-to-convert-list-of-objects-to-numpy-array/30012403
                for i in range(cdf_resolution):
                    index = max(len(comp) * (i + 1) // cdf_resolution - 1, 0) # Don't want negative index
                    # Update cdfs with the firm-level cdf
                    cdfs[fid, i] = comp[index]

        if self.reference_dict['fid'] == ['f1i', 'f2i']: # Unstack Event Study
            # Drop rows that were appended earlier and rename columns
            data = data[data['f2i'] >= 0]
            data = data.rename({'fid': 'f1i', 'comp': 'y1'}, axis=1)

        return cdfs, sorted(fids)

    def cluster(self, user_cluster={}):
        '''
        Cluster data and assign a new column giving the cluster for each firm.

        Arguments:
            user_cluster (dict): dictionary of parameters for clustering

                Dictionary parameters:

                    cdf_resolution (int): how many values to use to approximate the cdf

                    grouping (str): how to group the cdfs ('quantile_all' to get quantiles from entire set of data, then have firm-level values between 0 and 1; 'quantile_firm_small' to get quantiles at the firm-level and have values be compensations if small data; 'quantile_firm_large' to get quantiles at the firm-level and have values be compensations if large data, note that this is up to 50 times slower than 'quantile_firm_small' and should only be used if the dataset is too large to copy into a dictionary)

                    stayers_movers (str or None): if None, uses entire dataset; if 'stayers', uses only stayers; if 'movers', uses only movers

                    year (int or None): if None, uses entire dataset; if int, gives year of data to consider. Works only if data formatted as BipartiteLong

                    dropna (bool): if True, drop observations where firms aren't clustered; if False, keep all observations

                    user_KMeans (dict): use parameters defined in KMeans_dict for KMeans estimation (for more information on what parameters can be used, visit https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html), and use default parameters defined in class attribute default_KMeans for any parameters not specified

        Returns:
            frame (BipartiteLong): BipartiteLong with clusters
        '''
        frame = self.copy()

        # Update dictionary
        cluster_params = update_dict(frame.default_cluster, user_cluster)

        # Unpack dictionary
        cdf_resolution = cluster_params['cdf_resolution']
        grouping = cluster_params['grouping']
        year = cluster_params['year']
        stayers_movers = cluster_params['stayers_movers']
        user_KMeans = cluster_params['user_KMeans']

        # Compute cdfs
        cdfs, fids = frame.approx_cdfs(cdf_resolution=cdf_resolution, grouping=grouping, stayers_movers=stayers_movers, year=year)
        frame.logger.info('firm cdfs computed')

        # Compute firm clusters
        KMeans_params = update_dict(frame.default_KMeans, user_KMeans)
        clusters = KMeans(**KMeans_params).fit(cdfs).labels_
        frame.logger.info('firm clusters computed')
        
        # Drop existing clusters
        if frame.col_included('j'):
            frame.drop('j')

        for i, fid_col in enumerate(to_list(self.reference_dict['fid'])):
            if self.reference_dict['fid'] == 'fid':
                j_col = 'j'
            else:
                j_col = 'j' + str(i + 1)
            clusters_dict = {fid_col: fids, j_col: clusters}
            clusters_df = pd.DataFrame(clusters_dict, index=np.arange(len(fids)))
            frame.logger.info('dataframe linking fids to clusters generated')

            # Merge into event study data
            frame = frame.merge(clusters_df, how='left', on=fid_col)
            # Keep column as int even with nans
            frame[j_col] = frame[j_col].astype('Int64')
            frame.col_dict[j_col] = j_col

        # Sort columns
        sorted_cols = sorted(frame.columns, key=col_order)
        frame = frame[sorted_cols]

        if cluster_params['dropna']:
            # Drop firms that don't get clustered
            frame = frame.dropna().reset_index(drop=True)
            frame[self.reference_dict['j']] = frame[self.reference_dict['j']].astype(int)
            frame.clean_data()

        frame.contiguous_cids = True

        frame.logger.info('clusters merged into data')

        return frame
