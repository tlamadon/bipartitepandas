'''
Class for a bipartite network
'''
# from pandas.core.indexes.base import InvalidIndexError
from multiprocessing.sharedctypes import Value
import numpy as np
import pandas as pd
from pandas import DataFrame
import warnings
import bipartitepandas as bpd
from bipartitepandas.util import update_dict, to_list
import igraph as ig

def _recollapse_loop(force=False):
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

            if force or (isinstance(self, (bpd.BipartiteLongCollapsed, bpd.BipartiteEventStudyCollapsed)) and (not self.no_returns)):
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

# Define default parameter dictionaries
_clean_params_default = bpd.util.ParamsDict({
    'connectedness': (None, 'set', ['connected', 'leave_one_observation_out', 'leave_one_worker_out', 'leave_one_firm_out', None],
        '''
            (default=None) When computing largest connected set of firms: if 'connected', keep observations in the largest connected set of firms; if 'leave_one_observation_out', keep observations in the largest leave-one-observation-out connected set; if 'leave_one_worker_out', keep observations in the largest leave-one-worker-out connected set; if 'leave_one_firm_out', keep observations in the largest leave-one-firm-out connected set; if None, keep all observations.
        ''', None),
    'component_size_variable': ('firms', 'set', ['len', 'length', 'firms', 'workers', 'stayers', 'movers'],
        '''
        (default='firms') How to determine largest connected component. Options are 'len'/'length' (length of frame), 'firms' (number of unique firms), 'workers' (number of unique workers), 'stayers' (number of unique stayers), and 'movers' (number of unique movers).
        ''', None),
    'i_t_how': ('max', 'type', (*bpd.util.fn_type, str),
        '''
            (default='max') When dropping i-t duplicates: if 'max', keep max paying job; otherwise, take `how` over duplicate worker-firm-year observations, then take the highest paying worker-firm observation. `how` can take any input valid for a Pandas transform. Note that if multiple time and/or firm columns are included (as in collapsed long and event study formats), then data is converted to long, cleaned, then converted back to its original format.
        ''', None),
    'drop_returns': (False, 'set', [False, 'returns', 'returners', 'keep_first_returns', 'keep_last_returns'],
        '''
            (default=False) If 'returns', drop observations where workers leave a firm then return to it; if 'returners', drop workers who ever leave then return to a firm; if 'keep_first_returns', keep first spell where a worker leaves a firm then returns to it; if 'keep_last_returns', keep last spell where a worker leaves a firm then returns to it; if False, keep all observations.
        ''', None),
    'drop_returns_to_stays': (False, 'type', bool,
        '''
            (default=False) Applies only if 'drop_returns' is set to False. If True, when recollapsing collapsed data, drop observations that need to be recollapsed instead of collapsing (this is for computational efficiency when re-collapsing data for leave-one-out connected components, where intermediate observations can be dropped, causing a worker who returns to a firm to become a stayer).
        ''', None),
    'is_sorted': (False, 'type', bool,
        '''
            (default=False) If False, dataframe will be sorted by i (and t, if included). Set to True if already sorted.
        ''', None),
    'force': (True, 'type', bool,
        '''
            (default=True) If True, force all cleaning methods to run; much faster if set to False.
        ''', None),
    'copy': (True, 'type', bool,
        '''
            (default=True) If False, avoid copying data when possible.
        ''', None),
    'verbose': (True, 'type', bool,
        '''
            (default=True) If True, print progress during data cleaning.
        ''', None)
})

def clean_params(update_dict={}):
    '''
    Dictionary of default clean_params. Run bpd.clean_params().describe_all() for descriptions of all valid parameters.

    Arguments:
        update_dict (dict): user parameter values

    Returns:
        (ParamsDict) dictionary of clean_params
    '''
    new_dict = _clean_params_default.copy()
    new_dict.update(update_dict)
    return new_dict

_cluster_params_default = bpd.util.ParamsDict({
    'measures': (bpd.measures.CDFs(), 'list_of_type', (bpd.measures.CDFs, bpd.measures.Moments),
        '''
            (default=bpd.measures.CDFs()) How to compute measures for clustering. Options can be seen in bipartitepandas.measures.
        ''', None),
    'grouping': (bpd.grouping.KMeans(), 'type', (bpd.grouping.KMeans, bpd.grouping.Quantiles),
        '''
            (default=bpd.grouping.KMeans()) How to group firms based on measures. Options can be seen in bipartitepandas.grouping.
        ''', None),
    'stayers_movers': (None, 'set', [None, 'stayers', 'movers'],
        '''
            (default=None) If None, clusters on entire dataset; if 'stayers', clusters on only stayers; if 'movers', clusters on only movers.
        ''', None),
    't': (None, 'type_none', int,
        '''
            (default=None) If None, clusters on entire dataset; if int, gives period in data to consider (only valid for non-collapsed data).
        ''', None),
    'weighted': (True, 'type', bool,
        '''
            (default=True) If True, weight firm clusters by firm size (if a weight column is included, firm weight is computed using this column; otherwise, each observation is given weight 1).
        ''', None),
    'dropna': (False, 'type', bool,
        '''
            (default=False) If True, drop observations where firms aren't clustered; if False, keep all observations.
        ''', None),
    'clean_params': (None, 'type_none', bpd.util.ParamsDict,
        '''
            (default=None) Dictionary of parameters for cleaning. This is used when observations get dropped because they were not clustered. Default is None, which sets connectedness to be the connectedness measure previously used. Run bpd.clean_params().describe_all() for descriptions of all valid parameters.
        ''', None),
    'is_sorted': (False, 'type', bool,
        '''
            (default=False) For event study format. If False, dataframe will be sorted by i (and t, if included). Set to True if already sorted.
        ''', None),
    'copy': (True, 'type', bool,
        '''
            (default=True) If False, avoid copy.
        ''', None)
})

def cluster_params(update_dict={}):
    '''
    Dictionary of default cluster_params. Run bpd.cluster_params().describe_all() for descriptions of all valid parameters.

    Arguments:
        update_dict (dict): user parameter values

    Returns:
        (ParamsDict) dictionary of cluster_params
    '''
    new_dict = _cluster_params_default.copy()
    new_dict.update(update_dict)
    return new_dict

class BipartiteBase(DataFrame):
    '''
    Base class for BipartitePandas, where BipartitePandas gives a bipartite network of firms and workers. Contains generalized methods. Inherits from DataFrame.

    Arguments:
        *args: arguments for Pandas DataFrame
        columns_req (list): required columns (only put general column names for joint columns, e.g. put 'j' instead of 'j1', 'j2'; then put the joint columns in col_reference_dict)
        columns_opt (list): optional columns (only put general column names for joint columns, e.g. put 'g' instead of 'g1', 'g2'; then put the joint columns in col_reference_dict)
        columns_contig (dictionary): columns requiring contiguous ids linked to boolean of whether those ids are contiguous, or None if column(s) not included, e.g. {'i': False, 'j': False, 'g': None} (only put general column names for joint columns)
        col_reference_dict (dict): clarify which joint columns are associated with a general column name, e.g. {'i': 'i', 'j': ['j1', 'j2']}
        col_dtype_dict (dict): link column to datatype, e.g. {'m': 'int'}
        col_collapse_dict (dict): how to collapse column (None indicates the column should be dropped), e.g. {'y': 'mean'}
        col_long_es_dict (dict): whether each column should split into two when converting from long to event study (None indicates the column should be dropped), e.g. {'y': True, 'm': None}
        include_id_reference_dict (bool): if True, create dictionary of Pandas dataframes linking original id values to contiguous id values
        log (bool): if True, will create log file(s)
        **kwargs: keyword arguments for Pandas DataFrame
    '''
    # Attributes, required for Pandas inheritance
    _metadata = ['columns_req', 'columns_opt', 'columns_contig', 'col_reference_dict', 'col_dtype_dict', 'col_collapse_dict', 'col_long_es_dict', 'id_reference_dict', 'connectedness', 'no_na', 'no_duplicates', 'i_t_unique', 'no_returns', '_log_on_indicator', '_log_level_fn_dict']

    def __init__(self, *args, columns_req=[], columns_opt=[], columns_contig=[], col_reference_dict={}, col_dtype_dict={}, col_collapse_dict={}, col_long_es_dict={}, include_id_reference_dict=False, log=True, **kwargs):
        # Initialize DataFrame
        super().__init__(*args, **kwargs)

        # Start logger
        bpd.util.logger_init(self)
        # Option to turn on/off logger
        self._log_on_indicator = log
        # self.log('initializing BipartiteBase object', level='info')

        if len(args) > 0 and isinstance(args[0], BipartiteBase):
            # Note that isinstance works for subclasses
            self._set_attributes(args[0], no_dict=False, include_id_reference_dict=include_id_reference_dict)
            # Update class attributes from the previous dataframe with parameter inputs
            self.columns_req = ['i', 'j', 'y'] + [column_req for column_req in self.columns_req if column_req not in ['i', 'j', 'y']]
            self.columns_opt = ['t', 'g', 'w', 'm'] + [column_opt for column_opt in self.columns_opt if column_opt not in ['t', 'g', 'w', 'm']]
            self.columns_contig = update_dict(self.columns_contig, columns_contig)
            self.col_reference_dict = update_dict(self.col_reference_dict, col_reference_dict)
            self.col_dtype_dict = update_dict(self.col_dtype_dict, col_dtype_dict)
            self.col_collapse_dict = update_dict(self.col_collapse_dict, col_collapse_dict)
            self.col_long_es_dict = update_dict(self.col_long_es_dict, col_long_es_dict)
        else:
            self.columns_req = ['i', 'j', 'y'] + columns_req
            self.columns_opt = ['t', 'g', 'w', 'm'] + columns_opt
            self.columns_contig = update_dict({'i': False, 'j': False, 'g': None}, columns_contig)
            self.col_reference_dict = update_dict({'i': 'i', 'm': 'm'}, col_reference_dict)
            self.col_dtype_dict = update_dict({'i': 'any', 'j': 'any', 'y': 'float', 't': 'int', 'g': 'any', 'w': 'float', 'm': 'int'}, col_dtype_dict)
            # Skip t and m for collapsing
            self.col_collapse_dict = update_dict({'i': 'first', 'j': 'first', 'y': 'mean', 'g': 'first', 'w': 'sum'}, col_collapse_dict)
            # Split i to make sure consecutive observations are for the same worker
            self.col_long_es_dict = update_dict({'i': True, 'j': True, 'y': True, 't': True, 'g': True, 'w': True, 'm': False}, col_long_es_dict)

            # Link original id values to contiguous id values
            self._reset_id_reference_dict(include_id_reference_dict)

            # Set attributes
            self._reset_attributes()

        # Dictionary of logger functions based on level
        self._log_level_fn_dict = {
            'debug': self.logger.debug,
            'info': self.logger.info,
            'warning': self.logger.warning,
            'error': self.logger.error,
            'critical': self.logger.critical
        }

        # self.log('BipartiteBase object initialized', level='info')

    @property
    def _constructor(self):
        '''
        For inheritance from Pandas.
        '''
        return BipartiteBase

    def copy(self, deep=True):
        '''
        Return copy of self.

        Arguments:
            deep (bool): make a deep copy, including a copy of the data and the indices. If False, neither the indices nor the data are copied.

        Returns:
            (BipartiteBase): copy of dataframe
        '''
        self.log('beginning copy', level='info')
        df_copy = DataFrame(self).copy(deep=deep)
        # Set logging on/off depending on current selection
        bdf_copy = self._constructor(df_copy, log=self._log_on_indicator)
        # This copies attribute dictionaries, default copy does not
        bdf_copy._set_attributes(self)

        return bdf_copy

    def log_on(self, on=True):
        '''
        Toggle logger on or off.

        Arguments:
            on (bool): if True, turn logger on; if False, turn logger off
        '''
        self._log_on_indicator = on

    def log(self, message, level='info'):
        '''
        Log a message at the specified level.

        Arguments:
            message (str): message to log
            level (str): logger level. Options, in increasing severity, are 'debug', 'info', 'warning', 'error', and 'critical'.
        '''
        if self._log_on_indicator:
            # Log message
            self._log_level_fn_dict[level](message)

    def summary(self):
        '''
        Print summary statistics. This uses class attributes. To run a diagnostic to verify these values, run `.diagnostic()`.
        '''
        ret_str = ''
        y = self.loc[:, self.col_reference_dict['y']].to_numpy()
        mean_wage = np.mean(y)
        median_wage = np.median(y)
        max_wage = np.max(y)
        min_wage = np.min(y)
        var_wage = np.var(y)
        ret_str += f'format: {type(self).__name__!r}\n'
        ret_str += f'number of workers: {self.n_workers()}\n'
        ret_str += f'number of firms: {self.n_firms()}\n'
        ret_str += f'number of observations: {len(self)}\n'
        ret_str += f'mean wage: {mean_wage}\n'
        ret_str += f'median wage: {median_wage}\n'
        ret_str += f'min wage: {min_wage}\n'
        ret_str += f'max wage: {max_wage}\n'
        ret_str += f'var(wage): {var_wage}\n'
        ret_str += f'no NaN values: {self.no_na}\n'
        ret_str += f'no duplicates: {self.no_duplicates}\n'
        ret_str += f'i-t (worker-year) observations unique (None if t column(s) not included): {self.i_t_unique}\n'
        ret_str += f'no returns (None if not yet computed): {self.no_returns}\n'
        for contig_col, is_contig in self.columns_contig.items():
            ret_str += f'contiguous {contig_col!r} ids (None if not included): {is_contig}\n'
        ret_str += f'connectedness (None if ignoring connectedness): {self.connectedness!r}'

        print(ret_str)

    def diagnostic(self):
        '''
        Run diagnostic and print diagnostic report.
        '''
        ret_str = '----- General Diagnostic -----\n'
        ##### Sorted by i (and t, if included) #####
        sort_order = ['i']
        if self._col_included('t'):
            # If t column
            sort_order.append(to_list(self.col_reference_dict['t'])[0])
        is_sorted = (self.loc[:, sort_order] == self.loc[:, sort_order].sort_values(sort_order)).to_numpy().all()

        ret_str += f'sorted by i (and t, if included): {is_sorted}\n'

        ##### No NaN values #####
        # Source: https://stackoverflow.com/a/29530601/17333120
        no_na = (not self.isnull().to_numpy().any())

        ret_str += f'no NaN values: {no_na}\n'

        ##### No duplicates #####
        # https://stackoverflow.com/a/50243108/17333120
        no_duplicates = (not self.duplicated().any())

        ret_str += f'no duplicates: {no_duplicates}\n'

        ##### i-t unique #####
        no_i_t_duplicates = (not self.duplicated(subset=sort_order).any())

        ret_str += f'i-t (worker-year) observations unique (if t column(s) not included, then i observations unique): {no_i_t_duplicates}\n'

        ##### No returns #####
        no_returns = (len(self) == len(self._drop_returns(how='returns', reset_index=False)))
        ret_str += f'no returns: {no_returns}\n'

        ##### Contiguous ids #####
        for contig_col in self.columns_contig.keys():
            if self._col_included(contig_col):
                contig_ids = self.unique_ids(contig_col)
                is_contig = (len(contig_ids) == (max(contig_ids) + 1))
                ret_str += f'contiguous {contig_col!r} ids (None if not included): {is_contig}\n'
            else:
                ret_str += f'contiguous {contig_col!r} ids (None if not included): None\n'

        ##### Connectedness #####
        is_connected_dict = {
            None: lambda : None,
            'connected': lambda : self._construct_graph(self.connectedness, is_sorted=False, copy=False)[0].is_connected(),
            'leave_one_observation_out': lambda: (len(self) == len(self._conset(connectedness=self.connectedness, is_sorted=False))),
            'leave_one_firm_out': lambda: (len(self) == len(self._conset(connectedness=self.connectedness, is_sorted=False)))
        }
        is_connected = is_connected_dict[self.connectedness]()

        if is_connected or (is_connected is None):
            ret_str += f'frame connectedness is (None if ignoring connectedness): {self.connectedness!r}\n'
        else:
            ret_str += f'frame failed connectedness: {self.connectedness!r}\n'

        if self._col_included('m'):
            ##### m column #####
            m_correct = (self.loc[:, 'm'] == self.gen_m(force=True).loc[:, 'm']).to_numpy().all()

            ret_str += f"'m' column correct (None if not included): {m_correct}\n"
        else:
            ret_str += "'m' column correct (None if not included): None"

        print(ret_str)

    def add_column(self, col_data, col_name, col_reference=None, is_contig=False, dtype='any', how_collapse='first', long_es_split=True, copy=True):
        '''
        Safe method for adding custom columns. Columns added with this method will be compatible with conversions between long, collapsed long, event study, and collapsed event study formats.

        Arguments:
            col_data (NumPy Array or Pandas Series or list of (NumPy Array or Pandas Series)): data for column, or list of data for columns
            col_name (str): general column name
            col_reference (str or list of str): if column has multiple subcolumns (e.g. firm ids are associated with the columns ['j1', 'j2']) this must be specified; otherwise, None will automatically default to the column name (plus a column number, if more than one column is listed) (e.g. firm ids are associated with the column 'j' if one column is included, or ['j1', 'j2'] if two columns are included)
            is_contig (bool): if True, column is contiguous
            dtype (str): column datatype, must be one of 'int', 'float', or 'any'
            how_collapse (function or str or None): how to collapse data at the worker-firm spell level, must be a valid input for Pandas groupby; if None, column will be dropped during collapse/uncollapse
            long_es_split (bool or None) if True, column should split into two when converting from long to event study; if None, column will be dropped when converting between (collapsed) long and (collapsed) event study formats
            copy (bool): if False, avoid copy

        Returns:
            (BipartiteBase): dataframe with new column(s)
        '''
        if copy:
            frame = self.copy()
        else:
            frame = self

        # Before modifying anything, run a few checks
        if col_name in self.columns_req + self.columns_opt:
            # Check if column is a default column
            raise ValueError(f'Trying to add general column {col_name}, but this is reserved as a default column name. Default columns should be added using standard column assignment.')

        if frame._col_included(col_name):
            # Check if column already included
            raise ValueError(f'Trying to add general column {col_name}, but this column already exists.')

        col_data_lst = to_list(col_data)
        if col_reference is not None:
            # First, check that the length is correct
            if len(to_list(col_reference)) != len(col_data_lst):
                raise ValueError(f'Trying to add general column {col_name} with subcolumns {col_reference}, but while this reference includes {len(to_list(col_reference))} subcolumns, {len(col_data_lst)} columns of data were included as input.')

            # Next, check that no subcolumns are already included
            for subcol in to_list(col_reference):
                if subcol in self._included_cols(subcols=True):
                    raise ValueError(f'Trying to add subcolumn {subcol}, but this column already exists.')

            # If all checks pass, assign col_reference
            frame.col_reference_dict[col_name] = col_reference
        else:
            if len(col_data_lst) == 1:
                frame.col_reference_dict[col_name] = col_name
            else:
                frame.col_reference_dict[col_name] = [col_name + str(i + 1) for i in range(len(col_data_lst))]

        # Assign remaining class attributes
        if is_contig:
            frame.columns_contig[col_name] = None
            if frame.id_reference_dict:
                frame.id_reference_dict[col_name] = DataFrame()
        frame.col_dtype_dict[col_name] = dtype
        frame.col_collapse_dict[col_name] = how_collapse
        frame.col_long_es_dict[col_name] = long_es_split

        # Set data
        for i, subcol in enumerate(to_list(frame.col_reference_dict[col_name])):
            frame.loc[:, subcol] = col_data_lst[i]

        # Sort columns
        frame = frame.sort_cols(copy=False)

        return frame

    def unique_ids(self, id_col):
        '''
        Unique ids in column.

        Arguments:
            id_col (str): column to check ids ('i', 'j', or 'g'). Use general column names for joint columns, e.g. put 'j' instead of 'j1', 'j2'

        Returns:
            (NumPy Array or None): unique ids if column included; None otherwise
        '''
        self.log(f'finding unique ids in column {id_col!r}', level='info')
        if not self._col_included(id_col):
            # If column not in dataframe
            return None
        id_lst = []
        for id_subcol in to_list(self.col_reference_dict[id_col]):
            id_lst += list(self.loc[:, id_subcol].unique())
        return np.array(list(set(id_lst)))

    def n_unique_ids(self, id_col):
        '''
        Number of unique ids in column.

        Arguments:
            id_col (str): column to check ids ('i', 'j', or 'g'). Use general column names for joint columns, e.g. put 'j' instead of 'j1', 'j2'

        Returns:
            (int or None): number of unique ids if column included; None otherwise
        '''
        self.log(f'finding number of unique ids in column {id_col!r}', level='info')
        unique_ids = self.unique_ids(id_col)
        if unique_ids is None:
            return None
        return len(unique_ids)

    def n_workers(self):
        '''
        Get the number of unique workers.

        Returns:
            (int): number of unique workers
        '''
        self.log('finding unique workers', level='info')
        return self.loc[:, 'i'].nunique()

    def n_firms(self):
        '''
        Get the number of unique firms.

        Returns:
            (int): number of unique firms
        '''
        self.log('finding unique firms', level='info')
        return self.n_unique_ids('j')

    def n_clusters(self):
        '''
        Get the number of unique clusters.

        Returns:
            (int or None): (int or None): number of unique clusters if cluster column included; None otherwise
        '''
        self.log('finding unique clusters', level='info')
        return self.n_unique_ids('g')

    def original_ids(self, copy=True):
        '''
        Return self merged with original column ids.

        Arguments:
            copy (bool): if False, avoid copy

        Returns:
            (BipartiteBase or None): copy of self merged with original column ids, or None if id_reference_dict is empty
        '''
        self.log('returning self merged with original column ids', level='info')
        frame = DataFrame(self, copy=copy)
        if self.id_reference_dict:
            for id_col, reference_df in self.id_reference_dict.items():
                if len(reference_df) > 0:
                    # Make sure non-empty
                    for id_subcol in to_list(self.col_reference_dict[id_col]):
                        try:
                            frame = frame.merge(reference_df.loc[:, ['original_ids', f'adjusted_ids_{len(reference_df.columns) - 1}']].rename({'original_ids': f'original_{id_subcol}', f'adjusted_ids_{len(reference_df.columns) - 1}': id_subcol}, axis=1), how='left', on=id_subcol)
                        except TypeError:
                            # Int64 error with NaNs
                            frame.loc[:, id_col] = frame.loc[:, id_col].astype('Int64', copy=False)
                            frame = frame.merge(reference_df.loc[:, ['original_ids', f'adjusted_ids_{len(reference_df.columns) - 1}']].rename({'original_ids': f'original_{id_subcol}', f'adjusted_ids_{len(reference_df.columns) - 1}': id_subcol}, axis=1), how='left', on=id_subcol)
                # else:
                #     # If no changes, just make original_id be the same as the current id
                #     for id_subcol in to_list(self.col_reference_dict[id_col]):
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
            self.col_reference_dict = frame.col_reference_dict.copy()
        # Required, even if no_dict
        self.columns_contig = frame.columns_contig.copy()
        self.col_dtype_dict = frame.col_dtype_dict.copy()
        self.col_collapse_dict = frame.col_collapse_dict.copy()
        self.col_long_es_dict = frame.col_long_es_dict.copy()
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
        ## Booleans
        # If False, not connected; if 'connected', all observations are in the largest connected set of firms; if 'leave_one_observation_out', observations are in the largest leave-one-observation-out connected set; if 'leave_one_firm_out', observations are in the largest leave-one-firm-out connected set; if None, connectedness ignored
        self.connectedness = frame.connectedness
        # If True, no NaN observations in the data
        self.no_na = frame.no_na
        # If True, no duplicate rows in the data
        self.no_duplicates = frame.no_duplicates
        # If True, each worker has at most one observation per period
        self.i_t_unique = frame.i_t_unique
        # If True, no workers who leave a firm then return to it
        self.no_returns = frame.no_returns

    def _reset_attributes(self, columns_contig=True, connected=True, no_na=True, no_duplicates=True, i_t_unique=True, no_returns=True):
        '''
        Reset class attributes conditions to be False/None.

        Arguments:
            columns_contig (bool): if True, reset self.columns_contig
            connected (bool): if True, reset self.connectedness
            no_na (bool): if True, reset self.no_na
            no_duplicates (bool): if True, reset self.no_duplicates
            i_t_unique (bool): if True, reset self.i_t_unique
            no_returns (bool): if True, reset self.no_returns

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
            # If False, not connected; if 'connected', all observations are in the largest connected set of firms; if 'leave_one_observation_out', observations are in the largest leave-one-observation-out connected set; if 'leave_one_firm_out', observations are in the largest leave-one-firm-out connected set; if None, connectedness ignored
            self.connectedness = None
        if no_na:
            # If True, no NaN observations in the data
            self.no_na = False
        if no_duplicates:
            # If True, no duplicate rows in the data
            self.no_duplicates = False
        if i_t_unique:
            # If True, each worker has at most one observation per period; if None, t column not included (set to False later in method if t column included)
            self.i_t_unique = None

            # Verify whether period included
            if self._col_included('t'):
                self.i_t_unique = False
        if no_returns:
            # If True, no workers who leave a firm then return to it
            self.no_returns = None

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
            self.id_reference_dict = {id_col: DataFrame() for id_col in self.col_reference_dict.keys()}
        else:
            self.id_reference_dict = {}

        return self

    def _col_included(self, col):
        '''
        Check whether a column from the pre-established required/optional lists is included.

        Arguments:
            col (str): column to check. Use general column names for joint columns, e.g. put 'j' instead of 'j1', 'j2'

        Returns:
            (bool): True if column is included
        '''
        if col not in self.col_reference_dict.keys():
            return False
        for subcol in to_list(self.col_reference_dict[col]):
            if subcol not in self.columns:
                return False
        return True

    def _included_cols(self, subcols=False):
        '''
        Get all columns included from the pre-established required/optional lists.
        
        Arguments:
            subcols (bool): if False, uses general column names for joint columns, e.g. returns 'j' instead of 'j1', 'j2'

        Returns:
            all_cols (list): included columns
        '''
        all_cols = []
        for col, col_subcols in self.col_reference_dict.items():
            # Iterate through all columns
            if self._col_included(col):
                if subcols:
                    all_cols += to_list(col_subcols)
                else:
                    all_cols.append(col)
        return all_cols

    def drop(self, labels, axis=0, inplace=False, allow_optional=False, allow_required=False, **kwargs):
        '''
        Drop labels along axis.

        Arguments:
            labels (int or str, optionally as a list): row(s) or column(s) to drop. For columns, use general column names for joint columns, e.g. put 'g' instead of 'g1', 'g2'. Only user-added columns may be dropped, unless allow_optional or allow_required is set to True.
            axis (int or str): whether to drop labels from the 'index' (0) or 'columns' (1)
            inplace (bool): if True, modify in-place
            allow_optional (bool): if True, allow to drop optional columns
            allow_required (bool): if True, allow to drop required columns
            **kwargs: keyword arguments for Pandas drop

        Returns:
            frame (BipartiteBase): BipartiteBase with dropped labels
        '''
        frame = self

        if axis in [0, 'index']:
            if inplace:
                DataFrame.drop(frame, labels, axis=0, inplace=True, **kwargs)
            else:
                frame = DataFrame.drop(frame, labels, axis=0, inplace=False, **kwargs)
            # Since rows dropped, many properties might change
            frame._reset_attributes(no_na=False, no_duplicates=False, i_t_unique=False, no_returns=False)
        elif axis in [1, 'columns']:
            for col in to_list(labels):
                ## Start by checking if column is in col_reference_dict ##
                general_subcols_to_drop = []
                if col in frame.columns_req:
                    # If column required
                    if allow_required:
                        # If required columns are allowed to be dropped
                        general_subcols_to_drop = to_list(frame.col_reference_dict[col])
                    else:
                        # If required columns are not allowed to be dropped
                        warnings.warn(f'{col!r} is a required column and cannot be dropped without specifying allow_required=True. The returned frame has not dropped this column.')
                elif col in frame.columns_opt:
                    # If column optional
                    if allow_optional:
                        # If optional columns are allowed to be dropped
                        general_subcols_to_drop = to_list(frame.col_reference_dict[col])
                    else:
                        pass
                        # warnings.warn(f'{col!r} is a pre-defined optional column and cannot be dropped without specifying allow_optional=True. The returned frame has not dropped this column.')
                elif col in frame._included_cols():
                    # If column is user-added
                    general_subcols_to_drop = to_list(frame.col_reference_dict[col])
                    # Remove column from attribute dictionaries
                    if col in frame.columns_contig.keys():
                        del frame.columns_contig[col]
                        if frame.id_reference_dict:
                            del frame.id_reference_dict[col]
                    del frame.col_reference_dict[col], frame.col_dtype_dict[col], frame.col_collapse_dict[col], frame.col_long_es_dict[col]
                elif col in frame._included_cols(subcols=True):
                    # If column is a subcolumn
                    warnings.warn(f'{col!r} is a subcolumn. For columns listed in df.col_reference_dict, BipartitePandas only allows for general column names to be dropped. The returned frame has not dropped this column.')
                ## Now, check if column included, but is not in col_reference_dict ##
                else:
                    # If column is not pre-established
                    if col in frame.columns:
                        if inplace:
                            DataFrame.drop(frame, col, axis=1, inplace=True, **kwargs)
                        else:
                            frame = DataFrame.drop(frame, col, axis=1, inplace=False, **kwargs)
                    else:
                        raise ValueError(f'{col!r} is not in dataframe columns.')
                ## Finally, drop column if column is in col_reference_dict ##
                if len(general_subcols_to_drop) > 0:
                    # If dropping a column from col_reference_dict
                    for subcol in general_subcols_to_drop:
                        # Drop subcols
                        if inplace:
                            DataFrame.drop(frame, subcol, axis=1, inplace=True, **kwargs)
                        else:
                            frame = DataFrame.drop(frame, subcol, axis=1, inplace=False, **kwargs)
                    if col in frame.columns_contig.keys():
                        # If column contiguous
                        frame.columns_contig[col] = None
                        if frame.id_reference_dict:
                            # If id_reference_dict has been initialized, reset it for the dropped column
                            frame.id_reference_dict[col] = DataFrame()
        else:
            raise ValueError(f"Axis must be one of 0, 'index', 1, or 'columns'; input {axis} is invalid.")

        return frame

    def rename(self, rename_dict, axis=0, inplace=False, allow_optional=False, allow_required=False, **kwargs):
        '''
        Rename a column.

        Arguments:
            rename_dict (dict): key is current label, value is new label. When renaming columns, use general column names for joint columns, e.g. put 'g' instead of 'g1', 'g2'.
            axis (int or str): whether to drop labels from the 'index' (0) or 'columns' (1)
            inplace (bool): if True, modify in-place
            allow_optional (bool): if True, allow to rename optional columns
            allow_required (bool): if True, allow to rename required columns
            **kwargs: keyword arguments for Pandas rename

        Returns:
            frame (BipartiteBase): BipartiteBase with renamed labels
        '''
        frame = self

        if axis in [0, 'index']:
            if inplace:
                DataFrame.rename(frame, rename_dict, axis=0, inplace=True, **kwargs)
            else:
                frame = DataFrame.rename(frame, rename_dict, axis=0, inplace=False, **kwargs)
        elif axis in [1, 'columns']:
            if len(rename_dict.keys()) != len(set(rename_dict.keys())):
                # Make sure rename_dict keys are unique
                raise ValueError(f'.rename() requires that rename_dict keys are unique. However, the input {rename_dict} gives non-unique keys.')

            if len(rename_dict.values()) != len(set(rename_dict.values())):
                # Make sure rename_dict values are unique
                raise ValueError(f'.rename() requires that rename_dict values are unique. However, the input {rename_dict} gives non-unique values.')

            for col_cur, col_new in rename_dict.items():
                # Make sure col_cur != col_new
                if col_cur == col_new:
                    raise ValueError(f'.rename() requires that keys in rename_dict are distinct from their associated values. However, the input gives the key-value pair where both the key and the value are equal to {col_cur!r}.')
                # Make sure you don't rename a column to have the same name as another column without also renaming the second column
                if (col_new in frame.col_reference_dict.keys()) and (col_new not in rename_dict.keys()):
                    raise ValueError(f'.rename() requires that if a column is renamed to have the same name as another column, the second column must also be renamed. However, {col_cur!r} is set to be renamed to {col_new!r}, but {col_new!r} is already a column and is not being renamed.')

            # Copy attribute dictionaries to update them
            columns_contig = frame.columns_contig.copy()
            col_reference_dict = frame.col_reference_dict.copy()
            col_dtype_dict = frame.col_dtype_dict.copy()
            col_collapse_dict = frame.col_collapse_dict.copy()
            col_long_es_dict = frame.col_long_es_dict.copy()
            id_reference_dict = frame.id_reference_dict.copy()

            # This is the dictionary that will actually be used to rename the columns
            rename_dict_subcols = {}

            for col_cur, col_new in rename_dict.items():
                ## Start by checking if column is in col_reference_dict ##
                general_rename = False
                if col_cur in frame.columns_req:
                    # If column required
                    general_rename = True
                    if (col_cur in columns_contig.keys()) and id_reference_dict:
                        id_reference_dict[col_cur] = DataFrame()
                    if not allow_required:
                        # If required columns are not allowed to be renamed
                        raise ValueError(f'{col_cur!r} is a required column and cannot be renamed without specifying allow_required=True.')
                elif col_cur in frame.columns_opt:
                    # If column optional
                    general_rename = True
                    if (col_cur in columns_contig.keys()) and id_reference_dict:
                        id_reference_dict[col_cur] = DataFrame()
                    if not allow_optional:
                        # If optional columns are not allowed to be renamed
                        raise ValueError(f'{col_cur!r} is a pre-defined optional column and cannot be renamed without specifying allow_optional=True.')
                elif col_cur in frame._included_cols():
                    # If column is user-added
                    general_rename = True
                    # Delete current column from attribute dictionaries
                    if col_cur in columns_contig.keys():
                        del columns_contig[col_cur]
                        if id_reference_dict:
                            del id_reference_dict[col_cur]
                    del col_reference_dict[col_cur], col_dtype_dict[col_cur], col_collapse_dict[col_cur], col_long_es_dict[col_cur]
                elif col_cur in frame._included_cols(subcols=True):
                    # If column is a subcolumn
                    raise ValueError(f'{col_cur!r} is a subcolumn. For columns listed in df.col_reference_dict, BipartitePandas only allows for general column names to be renamed.')
                ## Now, check if column included, but is not in col_reference_dict ##
                else:
                    # If column is not pre-established
                    if col_cur in frame.columns:
                        rename_dict_subcols[col_cur] = col_new
                    else:
                        raise ValueError(f'{col_cur!r} is not in dataframe columns.')
                ## Finally, update rename_dict_subcols if column is in col_reference_dict ##
                if general_rename:
                    # If renaming a column from col_reference_dict
                    id_reference = []
                    for subcol_cur in to_list(frame.col_reference_dict[col_cur]):
                        # Construct new column name and set value for rename dictionary
                        subcol_str, subcol_num = bpd.util._text_num_split(subcol_cur)
                        subcol_new = col_new + subcol_num
                        rename_dict_subcols[subcol_cur] = subcol_new
                        id_reference.append(subcol_new)
                    if len(id_reference) == 1:
                        # If list is length 1, extract first entry from list
                        id_reference = id_reference[0]

                    # Update attribute dictionaries
                    if col_cur in columns_contig.keys():
                        columns_contig[col_new] = frame.columns_contig[col_cur]
                        if id_reference_dict:
                            id_reference_dict[col_new] = frame.id_reference_dict[col_cur]
                    col_reference_dict[col_new] = id_reference
                    col_dtype_dict[col_new] = frame.col_dtype_dict[col_cur]
                    col_collapse_dict[col_new] = frame.col_collapse_dict[col_cur]
                    col_long_es_dict[col_new] = frame.col_long_es_dict[col_cur]

            if inplace:
                DataFrame.rename(frame, rename_dict_subcols, axis=1, inplace=True, **kwargs)
            else:
                frame = DataFrame.rename(frame, rename_dict_subcols, axis=1, inplace=False, **kwargs)

            # Sort columns
            frame = frame.sort_cols(copy=False)

            # Set new attribute dictionaries (note that this needs to be done at the end, otherwise renaming might fail after attribute dictionaries have been changed - but the new attribute dictionaries are still maintained, despite being incorrect for the dataframe as it hasn't been renamed)
            frame.columns_contig = columns_contig
            frame.col_reference_dict = col_reference_dict
            frame.col_dtype_dict = col_dtype_dict
            frame.col_collapse_dict = col_collapse_dict
            frame.col_long_es_dict = col_long_es_dict
            frame.id_reference_dict = id_reference_dict
        else:
            raise (f"Axis must be one of 0, 'index', 1, or 'columns'; input {axis} is invalid.")

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
        # Use correct constructor
        frame = self._constructor(frame, log=self._log_on_indicator)
        if kwargs['how'] == 'left':
            # Non-left merge could cause issues with data, by default resets attributes
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
        self.log(f'making {id_col!r} ids contiguous', level='info')
        if copy:
            frame = self.copy()
        else:
            frame = self

        cols = to_list(frame.col_reference_dict[id_col])
        n_cols = len(cols)
        n_rows = len(frame)
        all_ids = frame.loc[:, cols].to_numpy().reshape(n_cols * n_rows)
        # Source: https://stackoverflow.com/questions/16453465/multi-column-factorize-in-pandas
        factorized = pd.factorize(all_ids)

        # Quickly check whether ids need to be reset
        try:
            if max(factorized[1]) + 1 == len(factorized[1]):
                # ids are already contiguous
                frame.columns_contig[id_col] = True
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
                id_reference_df = DataFrame({'adjusted_ids_' + str(n_cols_id - 1): factorized[1], 'adjusted_ids_' + str(n_cols_id): np.arange(len(factorized[1]))}, index=np.arange(len(factorized[1]))).astype('Int64', copy=False)
                frame.id_reference_dict[id_col] = frame.id_reference_dict[id_col].merge(id_reference_df, how='left', on='adjusted_ids_' + str(n_cols_id - 1))

        # Sort columns
        frame = frame.sort_cols(copy=False)

        # ids are now contiguous
        frame.columns_contig[id_col] = True

        return frame

    def clean_data(self, clean_params=clean_params()):
        '''
        Clean data to make sure there are no NaN or duplicate observations, observations where workers leave a firm then return to it are removed, firms are connected by movers, and firm ids are contiguous.

        Arguments:
            clean_params (ParamsDict): dictionary of parameters for cleaning. Run bpd.clean_params().describe_all() for descriptions of all valid parameters.

        Returns:
            frame (BipartiteBase): BipartiteBase with cleaned data
        '''
        self.log('beginning BipartiteBase data cleaning', level='info')

        force = clean_params['force']
        verbose = clean_params['verbose']

        if clean_params['copy']:
            frame = self.copy()
        else:
            frame = self

        # First, check that required columns are included and datatypes are correct
        frame.log('checking required columns and datatypes', level='info')
        if verbose:
            print('checking required columns and datatypes')
        frame._check_cols()

        # Next, sort rows
        frame.log('sorting rows', level='info')
        if verbose:
            print('sorting rows')
        frame = frame.sort_rows(is_sorted=clean_params['is_sorted'], copy=False)

        # Next, drop NaN observations
        if force or (not frame.no_na):
            frame.log('dropping NaN observations', level='info')
            if verbose:
                print('dropping NaN observations')
            if frame.isna().to_numpy().any():
                # Checking first is considerably faster if there are no NaN observations
                frame.dropna(inplace=True)

            # Update no_na
            frame.no_na = True

        # Generate 'm' column - this is necessary for the next steps
        # 'm' will get updated in the following steps as it changes
        frame.log("generating 'm' column", level='info')
        if verbose:
            print("generating 'm' column")
        frame = frame.gen_m(force=True, copy=False)

        # Next, make sure i-t (worker-year) observations are unique
        if (force or (not frame.i_t_unique)) and (frame.i_t_unique is not None):
            frame.log(f"keeping highest paying job for i-t (worker-year) duplicates (how={clean_params['i_t_how']!r})", level='info')
            if verbose:
                print(f"keeping highest paying job for i-t (worker-year) duplicates (how={clean_params['i_t_how']!r})")
            frame = frame._drop_i_t_duplicates(how=clean_params['i_t_how'], is_sorted=True, copy=False)

            # Update no_duplicates
            frame.no_duplicates = True
        elif force or (not frame.no_duplicates):
            # Drop duplicate observations
            frame.log('dropping duplicate observations', level='info')
            if verbose:
                print('dropping duplicate observations')
            frame.drop_duplicates(inplace=True)

            # Update no_duplicates
            frame.no_duplicates = True

        # Next, drop returns
        if force or (frame.no_returns is None) or ((not frame.no_returns) and clean_params['drop_returns']):
            frame.log(f"dropping workers who leave a firm then return to it (how={clean_params['drop_returns']!r})", level='info')
            if verbose:
                print(f"dropping workers who leave a firm then return to it (how={clean_params['drop_returns']!r})")
            frame = frame._drop_returns(how=clean_params['drop_returns'], is_sorted=True, reset_index=True, copy=False)

        # Next, check contiguous ids before using igraph (igraph resets ids to be contiguous, so we need to make sure ours are comparable)
        for contig_col, is_contig in frame.columns_contig.items():
            if frame._col_included(contig_col) and (force or (not is_contig)):
                frame.log(f'making {contig_col!r} ids contiguous', level='info')
                if verbose:
                    print(f'making {contig_col!r} ids contiguous')
                frame = frame._contiguous_ids(id_col=contig_col, copy=False)

        # Next, find largest set of firms connected by movers
        if force or (frame.connectedness in [False, None]):
            # Generate largest connected set
            frame.log(f"computing largest connected set (how={clean_params['connectedness']!r})", level='info')
            if verbose:
                print(f"computing largest connected set (how={clean_params['connectedness']!r})")
            frame = frame._conset(connectedness=clean_params['connectedness'], component_size_variable=clean_params['component_size_variable'], drop_returns_to_stays=clean_params['drop_returns_to_stays'], is_sorted=True, copy=False)

            # Next, check contiguous ids after igraph, in case the connected components dropped ids (_conset() automatically updates contiguous attributes)
            for contig_col, is_contig in frame.columns_contig.items():
                if frame._col_included(contig_col) and (not is_contig):
                    frame.log(f'making {contig_col!r} ids contiguous', level='info')
                    if verbose:
                        print(f'making {contig_col!r} ids contiguous')
                    frame = frame._contiguous_ids(id_col=contig_col, copy=False)

        # Sort columns
        frame.log('sorting columns', level='info')
        if verbose:
            print('sorting columns')
        frame = frame.sort_cols(copy=False)

        # Reset index
        frame.log('resetting index', level='info')
        if verbose:
            print('resetting index')
        frame.reset_index(drop=True, inplace=True)

        frame.log('data cleaning complete', level='info')
        if verbose:
            print('data cleaning complete')

        return frame

    def _check_cols(self):
        '''
        Check that required columns are included, that all columns have the correct datatype, and that all columns in the dataframe are listed in .col_reference_dict. Raises a ValueError if any of these checks fails.
        '''
        known_included_cols = []
        for col in self.col_reference_dict.keys():
            # Check all included referenced columns
            subcols = to_list(self.col_reference_dict[col])
            for subcol in subcols:
                # Check all joint columns
                if subcol not in self.columns:
                    # If column missing
                    error_msg = f'{subcol} missing from data'
                    self.log(error_msg, level='info')
                    if col in self.columns_req:
                        raise ValueError(error_msg)
                else:
                    # If column included, check type
                    if not bpd.util._is_subdtype(self.loc[:, subcol], self.col_dtype_dict[col]):
                        error_msg = f'{subcol} has the wrong dtype, it is currently {self.loc[:, subcol].dtype} but should be one of the following: {self.col_dtype_dict[col]}'
                        self.log(error_msg, level='info')
                        raise ValueError(error_msg)
                # If column included and has correct type, add to list
                known_included_cols.append(subcol)
        
        for col in self.columns:
            # Check all included columns (where some are possibly not referenced)
            if col not in known_included_cols:
                # If column not referenced
                error_msg = f"{col} is included in the dataframe but is not saved in .col_reference_dict. Please initialize your BipartiteBase object to include this column by setting 'col_reference_dict=your_col_reference_dict'."
                self.log(error_msg, level='info')
                raise ValueError(error_msg)

    def _conset(self, connectedness='connected', component_size_variable='firms', drop_returns_to_stays=False, is_sorted=False, copy=True):
        '''
        Update data to include only the largest connected component of movers.

        Arguments:
            connectedness (str or None): if 'connected', keep observations in the largest connected set of firms; if 'leave_one_observation_out', keep observations in the largest leave-one-observation-out connected set; if 'leave_one_worker_out', keep observations in the largest leave-one-worker-out connected set; if 'leave_one_firm_out', keep observations in the largest leave-one-firm-out connected set; if None, keep all observations
            component_size_variable (str): how to determine largest connected component. Options are 'len'/'length' (length of frame), 'firms' (number of unique firms), 'workers' (number of unique workers), 'stayers' (number of unique stayers), and 'movers' (number of unique movers).
            drop_returns_to_stays (bool): if True, when recollapsing collapsed data, drop observations that need to be recollapsed instead of collapsing (this is for computational efficiency when re-collapsing data for leave-one-out connected components, where intermediate observations can be dropped, causing a worker who returns to a firm to become a stayer)
            is_sorted (bool): if False, dataframe will be sorted by i (and t, if included). Set to True if already sorted.
            copy (bool): if False, avoid copy

        Returns:
            frame (BipartiteBase): BipartiteBase with connected component of movers
        '''
        if copy:
            frame = self.copy()
        else:
            frame = self

        frame.log(f'computing {connectedness!r} connected components (None if ignoring connectedness)', level='info')

        if connectedness is None:
            # Skipping connected set
            frame.connectedness = None
            # frame._check_contiguous_ids() # This is necessary
            frame.log(f'{connectedness!r} connected components (None if ignoring connectedness) computed', level='info')
            return frame

        # Keep track of whether contiguous ids change
        n_ids_prev = {id_col: frame.n_unique_ids(id_col) for id_col in frame.columns_contig}

        # Update data
        # Find largest connected set of firms
        # First, create graph
        G, max_j = frame._construct_graph(connectedness, is_sorted=is_sorted, copy=False)
        if connectedness == 'connected':
            # Compute all connected components of firms
            cc_list = sorted(G.components(), reverse=True, key=len)
            # Iterate over connected components to find the largest
            largest_cc = cc_list[0]
            frame_largest_cc = frame.keep_ids('j', largest_cc, is_sorted=is_sorted, copy=False)
            if component_size_variable != 'firms':
                # If component_size_varible is firms, no need to iterate
                for cc in cc_list[1:]:
                    frame_cc = frame.keep_ids('j', cc, is_sorted=True, copy=False)
                    replace = bpd.util.compare_frames(frame_largest_cc, frame_cc, size_variable=component_size_variable, operator='lt')
                    if replace:
                        frame_largest_cc = frame_cc
            frame = frame_largest_cc
            try:
                # Remove comp_size attribute
                del frame_largest_cc.comp_size
            except AttributeError:
                pass
        elif connectedness == 'leave_one_observation_out':
            if isinstance(frame, bpd.BipartiteEventStudyBase):
                warnings.warn('You should avoid computing leave-one-observation-out connected components on event study data. It requires converting data into long format and back into event study format, which is computationally expensive.')
            # Compute all connected components of firms (each entry is a connected component)
            cc_list = G.components()
            # Keep largest leave-one-observation-out set of firms
            frame = frame._leave_one_observation_out(cc_list=cc_list, max_j=max_j, component_size_variable=component_size_variable, drop_returns_to_stays=drop_returns_to_stays, is_sorted=is_sorted)
        elif connectedness == 'leave_one_worker_out':
            if isinstance(frame, bpd.BipartiteEventStudyBase):
                warnings.warn('You should avoid computing leave-one-worker-out connected components on event study data. It requires converting data into long format and back into event study format, which is computationally expensive.')
            # Compute all connected components of firms (each entry is a connected component)
            cc_list = G.components()
            # Keep largest leave-one-worker-out set of firms
            frame = frame._leave_one_worker_out(cc_list=cc_list, max_j=max_j, component_size_variable=component_size_variable, drop_returns_to_stays=drop_returns_to_stays, is_sorted=is_sorted)
        elif connectedness == 'leave_one_firm_out':
            if isinstance(frame, bpd.BipartiteEventStudyBase):
                warnings.warn('You should avoid computing leave-one-firm-out components on event study data. It requires converting data into long format and back into event study format, which is computationally expensive.')
            # Compute all biconnected components of firms (each entry is a biconnected component)
            bcc_list = G.biconnected_components()
            # Keep largest leave-one-firm-out set of firms
            frame = frame._leave_one_firm_out(bcc_list=bcc_list, component_size_variable=component_size_variable, drop_returns_to_stays=drop_returns_to_stays, is_sorted=is_sorted)
        else:
            raise NotImplementedError(f"Connectedness measure {connectedness!r} is invalid: it must be one of None, 'connected', 'leave_one_observation_out', 'leave_one_worker_out', or 'leave_one_firm_out'.")

        # Data is now connected
        frame.connectedness = connectedness

        # If number of ids changed, set contiguous to False
        for id_col in frame.columns_contig:
            if (n_ids_prev[id_col] is not None) and (n_ids_prev[id_col] != frame.n_unique_ids(id_col)):
                frame.columns_contig[id_col] = False

        frame.log(f'{connectedness!r} connected components (None if ignoring connectedness) computed', level='info')

        return frame

    def _construct_graph(self, connectedness='connected', is_sorted=False, copy=True):
        '''
        Construct igraph graph linking firms by movers.

        Arguments:
            connectedness (str): if 'connected', keep observations in the largest connected set of firms; if 'leave_one_observation_out', keep observations in the largest leave-one-observation-out connected set; if 'leave_one_worker_out', keep observations in the largest leave-one-worker-out connected set; if 'leave_one_firm_out', keep observations in the largest leave-one-firm-out connected set; if None, keep all observations
            is_sorted (bool): if False, dataframe will be sorted by i (and t, if included). Set to True if already sorted.
            copy (bool): if False, avoid copy

        Returns:
            (tuple of igraph Graph, int): (graph, maximum firm id)
        '''
        self.log(f'constructing {connectedness!r} graph', level='info')
        linkages_fn_dict = {
            'connected': self._construct_firm_linkages,
            'leave_one_observation_out': self._construct_firm_worker_linkages,
            'leave_one_worker_out': self._construct_firm_worker_linkages,
            'leave_one_firm_out': self._construct_firm_double_linkages
        }
        linkages, max_j = linkages_fn_dict[connectedness](is_sorted=is_sorted, copy=copy)
        # n_firms = self.loc[(self.loc[:, 'm'] > 0).to_numpy(), :].n_firms()
        return ig.Graph(edges=linkages), max_j # n=n_firms

    def sort_cols(self, copy=True):
        '''
        Sort frame columns (not in-place).

        Arguments:
            copy (bool): if False, avoid copy

        Returns:
            frame (BipartiteBase): BipartiteBase with columns sorted
        '''
        self.log('sorting columns', level='info')
        if copy:
            frame = self.copy()
        else:
            frame = self

        # Sort columns
        sorted_cols = bpd.util._sort_cols(frame.columns)
        frame = frame.reindex(sorted_cols, axis=1, copy=False)

        return frame

    def sort_rows(self, j_if_no_t=True, is_sorted=False, copy=True):
        '''
        Sort rows by i and t.

        Arguments:
            j_if_no_t (bool): if no time column, sort on j column instead
            is_sorted (bool): if False, dataframe will be sorted by i (and t, if included). Set to True if already sorted.
            copy (bool): if False, avoid copy

        Returns:
            frame (BipartiteBase): dataframe with rows sorted
        '''
        self.log('sorting rows', level='info')
        if copy:
            frame = self.copy()
        else:
            frame = self

        if not is_sorted:
            sort_order = ['i']
            if frame._col_included('t'):
                # If t column
                sort_order.append(to_list(frame.col_reference_dict['t'])[0])
            elif j_if_no_t:
                # If no t column, and choose to sort on j instead
                sort_order.append(to_list(frame.col_reference_dict['j'])[0])
            with bpd.util.ChainedAssignment():
                frame.sort_values(sort_order, inplace=True)

        return frame

    def drop_rows(self, rows, drop_returns_to_stays=False, is_sorted=False, reset_index=True, copy=True):
        '''
        Drop particular rows.

        Arguments:
            rows (list): rows to keep
            drop_returns_to_stays (bool): If True, when recollapsing collapsed data, drop observations that need to be recollapsed instead of collapsing (this is for computational efficiency when re-collapsing data for leave-one-out connected components, where intermediate observations can be dropped, causing a worker who returns to a firm to become a stayer)
            is_sorted (bool): if False, dataframe will be sorted by i (and t, if included). Set to True if already sorted.
            reset_index (bool): if True, reset index at end
            copy (bool): if False, avoid copy

        Returns:
            frame (BipartiteBase): dataframe with given rows dropped
        '''
        self.log('dropping rows', level='info')
        rows = set(rows)
        if len(rows) == 0:
            # If nothing input
            if copy:
                return self.copy()
            return self
        self_rows = set(self.index.to_numpy())
        rows_diff = self_rows.difference(rows)

        return self.keep_rows(rows_diff, drop_returns_to_stays=drop_returns_to_stays, is_sorted=is_sorted, reset_index=reset_index, copy=copy)

    def min_movers_firms(self, threshold=15, is_sorted=False, copy=True):
        '''
        List firms with at least `threshold` many movers.

        Arguments:
            threshold (int): minimum number of movers required to keep a firm
            is_sorted (bool): if False, dataframe will be sorted by i (and t, if included). Set to True if already sorted.
            copy (bool): if False, avoid copy

        Returns:
            valid_firms (NumPy Array): firms with sufficiently many movers
        '''
        self.log('computing firms with a minimum number of movers', level='info')
        if threshold == 0:
            # If no threshold
            return self.unique_ids('j')

        frame = self.loc[self.loc[:, 'm'].to_numpy() > 0, :]

        return frame.min_workers_firms(threshold, is_sorted=is_sorted, copy=copy)

    @_recollapse_loop(True)
    def min_movers_frame(self, threshold=15, drop_returns_to_stays=False, is_sorted=False, reset_index=True, copy=True):
        '''
        Return dataframe of firms with at least `threshold` many movers.

        Arguments:
            threshold (int): minimum number of movers required to keep a firm
            drop_returns_to_stays (bool): if True, when recollapsing collapsed data, drop observations that need to be recollapsed instead of collapsing (this is for computational efficiency when re-collapsing data for leave-one-out connected components, where intermediate observations can be dropped, causing a worker who returns to a firm to become a stayer)
            is_sorted (bool): if False, dataframe will be sorted by i (and t, if included). Set to True if already sorted.
            reset_index (bool): if True, reset index at end
            copy (bool): if False, avoid copy

        Returns:
            (BipartiteBase): dataframe of firms with sufficiently many movers
        '''
        self.log('generating frame of firms with a minimum number of movers', level='info')
        if threshold == 0:
            # If no threshold
            if copy:
                return self.copy()
            return self

        # FIXME this copy should not always be False, check out how to fix it
        valid_firms = self.min_movers_firms(threshold, is_sorted=is_sorted, copy=False)

        return self.keep_ids('j', keep_ids_list=valid_firms, drop_returns_to_stays=drop_returns_to_stays, is_sorted=is_sorted, reset_index=reset_index, copy=copy)

    def cluster(self, cluster_params=cluster_params()):
        '''
        Cluster data and assign a new column giving the cluster for each firm.

        Arguments:
            cluster_params (ParamsDict): dictionary of parameters for clustering. Run bpd.cluster_params().describe_all() for descriptions of all valid parameters.

        Returns:
            frame (BipartiteBase): BipartiteBase with clusters
        '''
        self.log('beginning clustering', level='info')
        if cluster_params['copy']:
            frame = self.copy()
        else:
            frame = self

        # Prepare data for clustering
        cluster_data, weights, jids = frame._prep_cluster(stayers_movers=cluster_params['stayers_movers'], t=cluster_params['t'], weighted=cluster_params['weighted'], is_sorted=cluster_params['is_sorted'], copy=False)

        # Compute measures
        for i, measure in enumerate(to_list(cluster_params['measures'])):
            if i == 0:
                computed_measures = measure.compute_measure(cluster_data, jids)
            else:
                # For computing both cdfs and moments
                computed_measures = np.concatenate([computed_measures, measure.compute_measure(cluster_data, jids)], axis=1)
        frame.log('firm moments computed', level='info')

        # Can't group using quantiles if more than 1 column
        if isinstance(cluster_params['grouping'], bpd.grouping.Quantiles) and (computed_measures.shape[1] > 1):
            raise NotImplementedError('Cannot cluster using quantiles if multiple measures computed.')

        # Compute firm groups
        frame.log('computing firm groups', level='info')
        clusters = cluster_params['grouping'].compute_groups(computed_measures, weights)
        frame.log('firm groups computed', level='info')

        # Link firms to clusters
        clusters_dict = dict(pd._lib.fast_zip([jids, clusters]))
        frame.log('dictionary linking firms to clusters generated', level='info')

        # Drop columns (because prepared data is not always a copy, must drop from self)
        for col in ['row_weights', 'one']:
            if col in self.columns:
                self.drop(col, axis=1, inplace=True)
            if col in frame.columns:
                frame.drop(col, axis=1, inplace=True)

        # Drop existing clusters
        if frame._col_included('g'):
            frame.drop('g', axis=1, inplace=True)

        for i, j_col in enumerate(to_list(frame.col_reference_dict['j'])):
            if len(to_list(frame.col_reference_dict['j'])) == 1:
                g_col = 'g'
            elif len(to_list(frame.col_reference_dict['j'])) == 2:
                g_col = 'g' + str(i + 1)

            # Merge into event study data
            frame[g_col] = frame[j_col].map(clusters_dict)
            # Keep column as int even with nans
            frame.loc[:, g_col] = frame.loc[:, g_col].astype('Int64', copy=False)

        # Sort columns
        frame = frame.sort_cols(copy=False)

        if cluster_params['dropna']:
            # Drop firms that don't get clustered
            frame.dropna(inplace=True)
            frame.reset_index(drop=True, inplace=True)
            frame.loc[:, frame.col_reference_dict['g']] = frame.loc[:, frame.col_reference_dict['g']].astype(int, copy=False)
            # Clean data
            if cluster_params['clean_params'] is None:
                frame = frame.clean_data(bpd.clean_params({'connectedness': frame.connectedness}))
            else:
                frame = frame.clean_data(cluster_params['clean_params'])

        frame.columns_contig['g'] = True

        frame.log('clusters merged into data', level='info')

        return frame
