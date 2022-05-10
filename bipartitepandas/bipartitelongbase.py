'''
Base class for bipartite networks in long or collapsed long format.
'''
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import bipartitepandas as bpd

class BipartiteLongBase(bpd.BipartiteBase):
    '''
    Base class for BipartiteLong and BipartiteLongCollapsed, where BipartiteLong and BipartiteLongCollapsed give a bipartite network of firms and workers in long and collapsed long form, respectively. Contains generalized methods. Inherits from BipartiteBase.

    Arguments:
        *args: arguments for BipartiteBase
        col_reference_dict (dict or None): clarify which columns are associated with a general column name, e.g. {'wid': 'wid', 'j': ['j1', 'j2']}; None is equivalent to {}
        **kwargs: keyword arguments for BipartiteBase
    '''

    def __init__(self, *args, col_reference_dict=None, **kwargs):
        # Update parameters to be lists/dictionaries instead of None (source: https://stackoverflow.com/a/54781084/17333120)
        if col_reference_dict is None:
            col_reference_dict = {}
        col_reference_dict = bpd.util.update_dict({'j': 'j', 'y': 'y', 'g': 'g', 'w': 'w'}, col_reference_dict)
        # Initialize DataFrame
        super().__init__(*args, col_reference_dict=col_reference_dict, **kwargs)

        # self.log('BipartiteLongBase object initialized', level='info')

    @property
    def _constructor(self):
        '''
        For inheritance from Pandas.

        Returns:
            (class): BipartiteLongBase class
        '''
        return BipartiteLongBase

    def gen_m(self, force=False, copy=True):
        '''
        Generate m column for data (m == 0 if stayer, m == 1 or 2 if mover).

        Arguments:
            force (bool): if True, reset 'm' column even if it exists
            copy (bool): if False, avoid copy

        Returns:
            (BipartiteLongBase): dataframe with m column
        '''
        if copy:
            frame = self.copy()
        else:
            frame = self

        if not frame._col_included('m') or force:
            i_col = frame.loc[:, 'i'].to_numpy()
            j_col = frame.loc[:, 'j'].to_numpy()
            i_prev = bpd.util.fast_shift(i_col, 1, fill_value=-2)
            i_next = bpd.util.fast_shift(i_col, -1, fill_value=-2)
            j_prev = np.roll(j_col, 1)
            j_next = np.roll(j_col, -1)

            with bpd.util.ChainedAssignment():
                frame.loc[:, 'm'] = ((i_col == i_prev) & (j_col != j_prev)).astype(int, copy=False) + ((i_col == i_next) & (j_col != j_next)).astype(int, copy=False)

            # Sort columns
            frame = frame.sort_cols(copy=False)

        else:
            frame.log("'m' column already included. Returning unaltered frame.", level='info')

        return frame

    def clean(self, params=None):
        '''
        Clean data to make sure there are no NaN or duplicate observations, observations where workers leave a firm then return to it are removed, firms are connected by movers, and categorical ids are contiguous.

        Arguments:
            params (ParamsDict or None): dictionary of parameters for cleaning. Run bpd.clean_params().describe_all() for descriptions of all valid parameters. None is equivalent to bpd.clean_params().

        Returns:
            (BipartiteLongBase): dataframe with cleaned data
        '''
        if params is None:
            params = bpd.clean_params()

        self.log('beginning BipartiteLongBase data cleaning', level='info')

        # Unpack parameters
        drop_returns = params['drop_returns']
        connectedness = params['connectedness']
        force = params['force']
        verbose = params['verbose']

        if params['copy']:
            frame = self.copy()
        else:
            frame = self

        # First, check that required columns are included and datatypes are correct
        self.log('checking required columns and datatypes', level='info')
        if verbose:
            tqdm.write('checking required columns and datatypes')
        frame._check_cols()

        # Next, sort rows
        self.log('sorting rows', level='info')
        if verbose:
            tqdm.write('sorting rows')
        frame = frame.sort_rows(is_sorted=params['is_sorted'], copy=False)

        # Next, drop NaN observations
        if force or (not frame.no_na):
            self.log('dropping NaN observations', level='info')
            if verbose:
                tqdm.write('dropping NaN observations')
            if frame.isna().to_numpy().any():
                # Checking first is considerably faster if there are no NaN observations
                frame.dropna(inplace=True)

            # Update no_na
            frame.no_na = True

        # Generate 'm' column - this is necessary for the next steps (note: 'm' will get updated in the following steps as it changes)
        self.log("generating 'm' column", level='info')
        if verbose:
            tqdm.write("generating 'm' column")
        frame = frame.gen_m(force=True, copy=False)

        # Next, make sure i-t (worker-year) observations are unique
        if (force or (not frame.i_t_unique)) and (frame.i_t_unique is not None):
            self.log(f"keeping highest paying job for i-t (worker-year) duplicates (how={params['i_t_how']!r})", level='info')
            if verbose:
                tqdm.write(f"keeping highest paying job for i-t (worker-year) duplicates (how={params['i_t_how']!r})")
            frame = frame._drop_i_t_duplicates(how=params['i_t_how'], is_sorted=True, copy=False)

            # Update no_duplicates
            frame.no_duplicates = True
        elif force or (not frame.no_duplicates):
            # Drop duplicate observations
            self.log('dropping duplicate observations', level='info')
            if verbose:
                tqdm.write('dropping duplicate observations')
            frame.drop_duplicates(inplace=True)

            # Update no_duplicates
            frame.no_duplicates = True

        # Next, drop returns
        if force or (frame.no_returns is None) or ((not frame.no_returns) and drop_returns):
            self.log(f"dropping workers who leave a firm then return to it (how={drop_returns!r})", level='info')
            if verbose:
                tqdm.write(f"dropping workers who leave a firm then return to it (how={drop_returns!r})")
            frame = frame._drop_returns(how=drop_returns, is_sorted=True, reset_index=True, copy=False)

        # Next, check categorical ids are contiguous before using igraph (igraph resets ids to be contiguous, so we need to make sure ours are comparable)
        for cat_col, is_contig in frame.columns_contig.items():
            if frame._col_included(cat_col) and (force or (not is_contig)):
                self.log(f'making {cat_col!r} ids contiguous', level='info')
                if verbose:
                    tqdm.write(f'making {cat_col!r} ids contiguous')
                frame = frame._make_categorical_contiguous(id_col=cat_col, copy=False)

        # Next, find largest set of firms connected by movers
        if force or (frame.connectedness in [False, None]):
            # Generate largest connected set
            self.log(f"computing largest connected set (how={connectedness!r})", level='info')
            if verbose:
                tqdm.write(f"computing largest connected set (how={connectedness!r})")
            frame = frame._connected_components(connectedness=connectedness, component_size_variable=params['component_size_variable'], drop_single_stayers=params['drop_single_stayers'], drop_returns_to_stays=params['drop_returns_to_stays'], is_sorted=True, copy=False)

            # Next, check categorical ids are contiguous after igraph, in case the connected components dropped ids (._connected_components() automatically updates contiguous attributes)
            for cat_col, is_contig in frame.columns_contig.items():
                if frame._col_included(cat_col) and (not is_contig):
                    self.log(f'making {cat_col!r} ids contiguous', level='info')
                    if verbose:
                        tqdm.write(f'making {cat_col!r} ids contiguous')
                    frame = frame._make_categorical_contiguous(id_col=cat_col, copy=False)

        # Sort columns
        self.log('sorting columns', level='info')
        if verbose:
            tqdm.write('sorting columns')
        frame = frame.sort_cols(copy=False)

        # Reset index
        self.log('resetting index', level='info')
        if verbose:
            tqdm.write('resetting index')
        frame.reset_index(drop=True, inplace=True)

        self.log('BipartiteLongBase data cleaning complete', level='info')

        return frame

    def to_eventstudy(self, move_to_worker=False, is_sorted=False, copy=True):
        '''
        Return (collapsed) long form data reformatted into (collapsed) event study data.

        Arguments:
            move_to_worker (bool): if True, each move is treated as a new worker
            is_sorted (bool): if False, dataframe will be sorted by i (and t, if included). Returned dataframe will be sorted. Sorting may alter original dataframe if copy is set to False. Set is_sorted to True if dataframe is already sorted.
            copy (bool): if False, avoid copy

        Returns:
            (BipartiteEventStudyBase): event study dataframe
        '''
        if not self._col_included('t'):
            raise NotImplementedError("Cannot convert from long to event study format without a time column. To bypass this, if you know your data is ordered by time but do not have time data, it is recommended to construct an artificial time column by calling .construct_artificial_time(copy=False).")

        # Sort and copy
        frame = self.sort_rows(is_sorted=is_sorted, copy=copy)

        # Split workers by movers and stayers
        worker_m = frame.get_worker_m(is_sorted=True)
        stayers = pd.DataFrame(frame.loc[~worker_m, :])
        movers = pd.DataFrame(frame.loc[worker_m, :])
        frame.log('workers split by movers and stayers', level='info')

        ## Add lagged values
        all_cols = frame._included_cols()
        default_cols = frame.columns_req + frame.columns_opt

        # Columns to keep
        keep_cols = ['i']
        # Keep track of user-added columns
        user_added_cols = {}
        for col in all_cols:
            if frame.col_long_es_dict[col] is None:
                # If None, drop this column
                pass
            elif frame.col_long_es_dict[col]:
                # If column should split
                for subcol in bpd.util.to_list(frame.col_reference_dict[col]):
                    # Get column number, e.g. j1 will give 1
                    subcol_number = subcol.strip(col)
                    ## Movers
                    # Useful for t1 and t2: t1 should go to t11 and t21; t2 should go to t12 and t22
                    col_1 = col + '1' + subcol_number
                    col_2 = col + '2' + subcol_number
                    # Lagged value
                    with bpd.util.ChainedAssignment():
                        movers.loc[:, col_1] = bpd.util.fast_shift(movers.loc[:, subcol].to_numpy(), 1, fill_value=-2)
                    movers.rename({subcol: col_2}, axis=1, inplace=True)

                    if subcol != 'i':
                        ## Stayers (no lags)
                        stayers.loc[:, col_1] = stayers.loc[:, subcol]
                        stayers.rename({subcol: col_2}, axis=1, inplace=True)

                        # Columns to keep
                        keep_cols += [col_1, col_2]
                        if col not in default_cols:
                            # User-added columns
                            if col in user_added_cols.keys():
                                user_added_cols[col] += [col_1, col_2]
                            else:
                                user_added_cols[col] = [col_1, col_2]
            else:
                # If column shouldn't split
                keep_cols += bpd.util.to_list(frame.col_reference_dict[col])
                if col not in default_cols:
                    # User-added columns
                    user_added_cols[col] = frame.col_reference_dict[col]

        # Ensure lagged values are for the same worker
        movers = movers.loc[movers.loc[:, 'i1'].to_numpy() == movers.loc[:, 'i2'].to_numpy(), :]

        # Set 'm' for movers
        movers.loc[:, 'm'] = 1
        movers.loc[movers.loc[:, 'j1'].to_numpy() == movers.loc[:, 'j2'].to_numpy(), 'm'] = 0

        # Correct datatypes (shifting adds nans which converts all columns into float, correct columns that should be int)
        for col in all_cols:
            if ((frame.col_dtype_dict[col] == 'int') or (col in frame.columns_contig.keys())) and frame.col_long_es_dict[col]:
                for subcol in bpd.util.to_list(frame.col_reference_dict[col]):
                    # Get column number, e.g. j1 will give 1
                    subcol_number = subcol.strip(col)
                    shifted_col = col + '1' + subcol_number
                    movers.loc[:, shifted_col] = movers.loc[:, shifted_col].astype(int, copy=False)

        # Correct i
        movers.drop('i2', axis=1, inplace=True)
        movers.rename({'i1': 'i'}, axis=1, inplace=True)

        # Keep only relevant columns
        stayers = stayers.reindex(keep_cols, axis=1, copy=False)
        movers = movers.reindex(keep_cols, axis=1, copy=False)
        frame.log('columns updated', level='info')

        # Merge stayers and movers (NOTE: this converts the data into a Pandas DataFrame)
        data_es = pd.concat([stayers, movers], ignore_index=True) # .reset_index(drop=True)

        # Sort columns
        sorted_cols = bpd.util._sort_cols(data_es.columns)
        data_es = data_es.reindex(sorted_cols, axis=1, copy=False)

        frame.log('data reformatted as event study', level='info')

        es_frame = frame._constructor_es(data_es, col_reference_dict=user_added_cols, log=frame._log_on_indicator)
        es_frame._set_attributes(frame, no_dict=True)

        for col, long_es_split in frame.col_long_es_dict.items():
            # Remove dropped columns from attribute dictionaries
            if long_es_split is None:
                # If column should be dropped during conversion to event study format
                del es_frame.col_dtype_dict[col]
                del es_frame.col_collapse_dict[col]
                del es_frame.col_long_es_dict[col]
                if col in es_frame.columns_contig.keys():
                    # If column is categorical
                    del es_frame.columns_contig[col]
                    if es_frame.id_reference_dict:
                        # If linking contiguous ids to original ids
                        del es_frame.id_reference_dict[col]

        # Sort data by i and t
        es_frame = es_frame.sort_rows(is_sorted=False, copy=False)

        # Reset index
        es_frame.reset_index(drop=True, inplace=True)

        if move_to_worker:
            es_frame.loc[:, 'i'] = es_frame.index

        return es_frame

    def _get_spell_ids(self, is_sorted=False, copy=True):
        '''
        Generate array of spell ids, where a spell is defined as an uninterrupted period of time where a worker works at the same firm. Spell ids are generated on sorted data, so it is recommended to sort your data using .sort_rows() prior to calling this method, then run the method with is_sorted=True.

        Arguments:
            is_sorted (bool): if False, dataframe will be sorted by i (and t, if included). Sorting may alter original dataframe if copy is set to False. Set is_sorted to True if dataframe is already sorted.
            copy (bool): if False, avoid copy

        Returns:
            (NumPy Array): spell ids
        '''
        self.log('preparing to compute spell ids', level='info')

        if not is_sorted:
            raise NotImplementedError('._get_spell_ids() requires `is_sorted` == True, but it is set to False.')

        # Sort and copy
        frame = self.sort_rows(is_sorted=is_sorted, copy=copy)
        self.log('data sorted by i (and t, if included)', level='info')

        # Introduce lagged i and j
        i_col = frame.loc[:, 'i'].to_numpy()
        j_col = frame.loc[:, 'j'].to_numpy()
        i_prev = bpd.util.fast_shift(i_col, 1, fill_value=-2)
        j_prev = np.roll(j_col, 1)
        self.log('lagged i and j introduced', level='info')

        # Generate spell ids (allow for i != i_prev to ensure that consecutive workers at the same firm get counted as different spells)
        # Source: https://stackoverflow.com/questions/59778744/pandas-grouping-and-aggregating-consecutive-rows-with-same-value-in-column
        new_spell = (j_col != j_prev) | (i_col != i_prev)
        del i_col, j_col, i_prev, j_prev

        spell_ids = new_spell.cumsum()
        self.log('spell ids generated', level='info')

        return spell_ids

    def _drop_returns(self, how=False, is_sorted=False, reset_index=True, copy=True):
        '''
        Drop observations where workers leave a firm then return to it.

        Arguments:
            how (str or False): if 'returns', drop observations where workers leave a firm then return to it; if 'returners', drop workers who ever leave then return to a firm; if 'keep_first_returns', keep first spell where a worker leaves a firm then returns to it; if 'keep_last_returns', keep last spell where a worker leaves a firm then returns to it; if False, keep all observations
            is_sorted (bool): if False, dataframe will be sorted by i (and t, if included). Returned dataframe will be sorted. Sorting may alter original dataframe if copy is set to False. Set is_sorted to True if dataframe is already sorted.
            reset_index (bool): if True, reset index at end
            copy (bool): if False, avoid copy

        Returns:
            (BipartiteLongBase): dataframe that drops observations where workers leave a firm then return to it
        '''
        self.log('preparing to drop returns', level='info')

        # Sort and copy
        frame = self.sort_rows(is_sorted=is_sorted, copy=copy)
        self.log('data sorted by i (and t, if included)', level='info')

        # Generate spell ids
        frame.loc[:, 'spell_id'] = frame._get_spell_ids(is_sorted=True, copy=False)

        # Find returns
        frame.loc[:, 'return_row'] = (frame.groupby(['i', 'j'], sort=False)['spell_id'].transform('nunique') > 1).astype(int, copy=False)

        # Check whether there are already no returns, or if we aren't dropping returns
        no_returns = (frame.loc[:, 'return_row'].sum() == 0)
        if no_returns or (not how):
            # Set frame.no_returns
            frame.no_returns = no_returns

            # Drop columns
            frame = frame.drop(['spell_id', 'return_row'], axis=1, inplace=True)

            return frame
        del no_returns

        if how == 'returners':
            # Find returners
            frame.loc[:, 'return_row'] = frame.groupby('i', sort=False)['return_row'].transform('max')
        elif how in ['keep_first_returns', 'keep_last_returns']:
            # Find the first/last spell in a return, and keep it
            frame_return_rows = frame.loc[frame.loc[:, 'return_row'].to_numpy() == 1, :]
            if how == 'keep_first_returns':
                return_spells_keep = frame_return_rows.groupby(['i', 'j'], sort=False)['spell_id'].first().unique()
            elif how == 'keep_last_returns':
                return_spells_keep = frame_return_rows.groupby(['i', 'j'], sort=False)['spell_id'].last().unique()
            # No longer mark first/last spells as return rows
            # frame.loc[frame.loc[:, 'spell_id'].isin(return_spells_keep), 'return_row'] = 0
            keep_rows = frame_return_rows.loc[frame_return_rows.loc[:, 'spell_id'].isin(return_spells_keep), :].index
            frame.loc[keep_rows, 'return_row'] = 0
            del frame_return_rows, return_spells_keep, keep_rows

        ## Drop returns
        # Find rows
        return_rows = np.where(frame.loc[:, 'return_row'].to_numpy() == 1)[0]

        # Drop columns (before drop rows)
        frame = frame.drop(['spell_id', 'return_row'], axis=1, inplace=True)

        # Drop returns
        frame = frame.drop_rows(return_rows, drop_returns_to_stays=False, is_sorted=True, reset_index=reset_index, copy=False)

        # Set frame.no_returns
        frame.no_returns = True

        self.log('returns dropped', level='info')

        return frame

    def _prep_cluster(self, stayers_movers=None, t=None, weighted=True, is_sorted=False, copy=False):
        '''
        Prepare data for clustering.

        Arguments:
            stayers_movers (str or None, default=None): if None, clusters on entire dataset; if 'stayers', clusters on only stayers; if 'movers', clusters on only movers; if 'stays', clusters on only stays; if 'moves', clusters on only moves
            t (int or list of int or None, default=None): if None, clusters on entire dataset; if int, gives period in data to consider (only valid for non-collapsed data); if list of int, gives periods in data to consider (only valid for non-collapsed data)
            weighted (bool, default=True): if True, weight firm clusters by firm size (if a weight column is included, firm weight is computed using this column; otherwise, each observation has weight 1)
            is_sorted (bool): if False, dataframe will be sorted by i in a groupby (but self will not be not sorted). Set is_sorted to True if dataframe is already sorted.
            copy (bool): if False, avoid copy
        Returns:
            data (Pandas DataFrame): data prepared for clustering
            weights (NumPy Array or None): if weighted=True, gives NumPy array of firm weights for clustering; otherwise, is None
            jids (NumPy Array): firm ids of firms in subset of data used to cluster
        '''
        if copy:
            frame = self.copy()
        else:
            frame = self

        if stayers_movers is not None:
            if stayers_movers == 'stayers':
                frame = frame.loc[~(frame.get_worker_m(is_sorted=is_sorted)), :]
            elif stayers_movers == 'movers':
                frame = frame.loc[frame.get_worker_m(is_sorted=is_sorted), :]
            elif stayers_movers == 'stays':
                frame = frame.loc[frame.loc[:, 'm'].to_numpy() == 0, :]
            elif stayers_movers == 'moves':
                frame = frame.loc[frame.loc[:, 'm'].to_numpy() > 0, :]
            else:
                raise NotImplementedError(f"Invalid 'stayers_movers' option, {stayers_movers!r}. Valid options are 'stayers', 'movers', 'stays', 'moves', or None.")

        # If period-level, then only use data for that particular period
        if t is not None:
            if isinstance(frame, bpd.BipartiteLong):
                t = bpd.util.to_list(t)
                if len(t) == 1:
                    frame = frame.loc[frame.loc[:, 't'].to_numpy() == t, :]
                else:
                    frame = frame.loc[frame.loc[:, 't'].isin(t), :]
            else:
                raise NotImplementedError("Cannot use data from a particular period with collapsed data. Data can be converted to long format using the .uncollapse() method.")

        with bpd.util.ChainedAssignment():
            # Create weights
            if weighted:
                if frame._col_included('w'):
                    frame.loc[:, 'row_weights'] = frame.loc[:, 'w']
                    weights = frame.groupby('j')['w'].sum().to_numpy()
                else:
                    frame.loc[:, 'row_weights'] = 1
                    weights = frame.groupby('j').size().to_numpy()
            else:
                frame.loc[:, 'row_weights'] = 1
                weights = None

        # Get unique firm ids (must sort)
        jids = np.sort(frame.loc[:, 'j'].unique())

        return frame, weights, jids

    def _get_articulation_matches(self, G, max_j):
        '''
        Compute articulation matches for self, by checking whether self is leave-one-match-out connected when dropping selected matches one at a time.

        Arguments:
            G (igraph Graph): graph linking firms by movers
            max_j (int): maximum j

        Returns:
            (NumPy Array): indices of articulation matches
        '''
        # Find bridges (recall i is adjusted to be greater than j, which is why we reverse the order) (source: https://igraph.discourse.group/t/function-to-find-edges-which-are-bridges-in-r-igraph/154/2)
        bridges = [tuple(sorted(a, reverse=True)) for a in G.biconnected_components() if len(a) == 2]
        bridges_workers = set([bridge[0] - (max_j + 1) for bridge in bridges])
        bridges_firms = set([bridge[1] for bridge in bridges])

        # Return articulation matches
        # return self.loc[pd.Series(map(tuple, self.loc[:, ['i', 'j']].to_numpy())).reindex_like(self, copy=False).isin(bridges), :].index.to_numpy() # FIXME this doesn't work
        return self.loc[self.loc[:, 'i'].isin(bridges_workers) & self.loc[:, 'j'].isin(bridges_firms), :].index.to_numpy()

    def _leave_out_observation_spell_match(self, cc_list, max_j, leave_out_group, component_size_variable='firms', drop_returns_to_stays=False, frame_largest_cc=None, is_sorted=False, copy=True, first_loop=True):
        '''
        Extract largest leave-one-(observation/spell/match)-out connected component.

        Arguments:
            cc_list (list of lists): each entry is a connected component
            max_j (int): maximum j in graph
            leave_out_group (str): which type of leave-one-out connected component to compute (options are 'observation', 'spell', or 'match')
            component_size_variable (str): how to determine largest leave-one-(observation/spell/match)-out connected component. Options are 'len'/'length' (length of frames), 'firms' (number of unique firms), 'workers' (number of unique workers), 'stayers' (number of unique stayers), 'movers' (number of unique movers), 'firms_plus_workers' (number of unique firms + number of unique workers), 'firms_plus_stayers' (number of unique firms + number of unique stayers), 'firms_plus_movers' (number of unique firms + number of unique movers), 'len_stayers'/'length_stayers' (number of stayer observations), 'len_movers'/'length_movers' (number of mover observations), 'stays' (number of stay observations), and 'moves' (number of move observations).
            drop_returns_to_stays (bool): if True, when recollapsing collapsed data, drop observations that need to be recollapsed instead of collapsing (this is for computational efficiency when re-collapsing data for leave-one-(observation/spell/match)-out connected components, where intermediate observations can be dropped, causing a worker who returns to a firm to become a stayer)
            frame_largest_cc (BipartiteLongBase): dataframe of baseline largest leave-one-(observation/spell/match)-out connected component
            is_sorted (bool): if False, dataframe will be sorted by i (and t, if included). Returned dataframe will be sorted. Sorting may alter original dataframe if copy is set to False. Set is_sorted to True if dataframe is already sorted.
            copy (bool): if False, avoid copy
            first_loop (bool): if True, this is the first loop of the method

        Returns:
            (BipartiteLongBase): dataframe of largest leave-one-(observation/spell/match)-out connected component
        '''
        # Sort and copy
        frame_init = self.sort_rows(is_sorted=is_sorted, copy=copy)

        for cc in sorted(cc_list, reverse=True, key=len):
            cc = np.array(cc)
            cc_j = cc[cc <= max_j]
            if frame_largest_cc is not None:
                # Can check if frame_cc is already smaller than frame_largest_cc before any computations
                skip = False
                if component_size_variable == 'firms':
                    # If looking at number of firms
                    try:
                        skip = (frame_largest_cc.comp_size >= len(cc_j))
                    except AttributeError:
                        frame_largest_cc.comp_size = frame_largest_cc.n_firms()
                        skip = (frame_largest_cc.comp_size >= len(cc_j))
                elif component_size_variable == 'firms_plus_movers':
                    # If looking at number of firms plus number of movers
                    try:
                        skip = (frame_largest_cc.comp_size >= len(cc))
                    except AttributeError:
                        frame_largest_cc.comp_size = frame_largest_cc.n_firms() + frame_largest_cc.loc[frame_largest_cc.loc[:, 'm'].to_numpy() > 0, :].n_workers()
                        skip = (frame_largest_cc.comp_size >= len(cc))
                if skip:
                    continue

            # Keep observations in connected components (NOTE: this does not require a copy)
            frame_cc = frame_init.keep_ids('j', cc_j, drop_returns_to_stays, is_sorted=True, copy=False)

            if frame_largest_cc is not None:
                # If frame_cc is already smaller than frame_largest_cc
                skip = bpd.util.compare_frames(frame_largest_cc, frame_cc, size_variable=component_size_variable, operator='geq', save_to_frame1=True, is_sorted=True)

                if skip:
                    continue

            # Remove firms with only 1 mover observation (can have 1 mover with multiple observations)
            frame_cc = frame_cc.min_moves_frame(2, drop_returns_to_stays, is_sorted=True, copy=False)

            if frame_largest_cc is not None:
                # If frame_cc is already smaller than frame_largest_cc
                skip = bpd.util.compare_frames(frame_largest_cc, frame_cc, size_variable=component_size_variable, operator='geq', save_to_frame1=True, is_sorted=True)

                if skip:
                    continue

            # Construct graph
            G2, max_j2 = frame_cc._construct_graph(f'leave_out_{leave_out_group}', is_sorted=True, copy=False)

            # Extract articulation rows
            articulation_fn_dict = {
                'observation': frame_cc._get_articulation_observations,
                'spell': frame_cc._get_articulation_spells,
                'match': frame_cc._get_articulation_matches
            }
            articulation_params_dict = {
                'observation': {'G': G2, 'max_j': max_j2, 'is_sorted': True},
                'spell': {'G': G2, 'max_j': max_j2, 'is_sorted': True, 'copy': False},
                'match': {'G': G2, 'max_j': max_j2}
            }
            articulation_rows = articulation_fn_dict[leave_out_group](**articulation_params_dict[leave_out_group])

            if len(articulation_rows) > 0:
                # If new frame is not leave-one-(observation/spell/match)-out connected, recompute connected components after dropping articulation rows (but note that articulation rows should be kept in the final dataframe) (NOTE: this does not require a copy)
                G2, max_j3 = frame_cc.drop_rows(articulation_rows, drop_returns_to_stays, is_sorted=True, reset_index=False, copy=False)._construct_graph(f'leave_out_{leave_out_group}', is_sorted=True, copy=False)
                cc_list_2 = G2.components()
                if len(cc_list_2) > 1:
                    # Recursion step (only necessary if dropping articulation workers disconnects the set of firms)
                    frame_cc = frame_cc._leave_out_observation_spell_match(cc_list=cc_list_2, max_j=max_j3, leave_out_group=leave_out_group, component_size_variable=component_size_variable, drop_returns_to_stays=drop_returns_to_stays, frame_largest_cc=frame_largest_cc, is_sorted=True, copy=False, first_loop=False)

            if frame_largest_cc is None:
                # If in the first round
                replace = True
            elif frame_cc is None:
                # If the biconnected components have recursively been eliminated
                replace = False
            else:
                replace = bpd.util.compare_frames(frame_largest_cc, frame_cc, size_variable=component_size_variable, operator='lt', save_to_frame1=True, is_sorted=True)
            if replace:
                frame_largest_cc = frame_cc

        if first_loop:
            # Remove comp_size attribute before return
            try:
                del frame_largest_cc.comp_size
            except AttributeError:
                pass

        # Return largest leave-one-(observation/spell/match)-out component
        return frame_largest_cc

    def _leave_out_worker(self, cc_list, max_j, component_size_variable='firms', drop_returns_to_stays=False, frame_largest_cc=None, is_sorted=False, copy=True, first_loop=True):
        '''
        Extract largest leave-one-worker-out connected component.

        Arguments:
            cc_list (list of lists): each entry is a connected component
            max_j (int): maximum j in graph
            component_size_variable (str): how to determine largest leave-one-worker-out connected component. Options are 'len'/'length' (length of frames), 'firms' (number of unique firms), 'workers' (number of unique workers), 'stayers' (number of unique stayers), 'movers' (number of unique movers), 'firms_plus_workers' (number of unique firms + number of unique workers), 'firms_plus_stayers' (number of unique firms + number of unique stayers), 'firms_plus_movers' (number of unique firms + number of unique movers), 'len_stayers'/'length_stayers' (number of stayer observations), 'len_movers'/'length_movers' (number of mover observations), 'stays' (number of stay observations), and 'moves' (number of move observations).
            drop_returns_to_stays (bool): if True, when recollapsing collapsed data, drop observations that need to be recollapsed instead of collapsing (this is for computational efficiency when re-collapsing data for leave-one-out connected components, where intermediate observations can be dropped, causing a worker who returns to a firm to become a stayer)
            frame_largest_cc (BipartiteLongBase): dataframe of baseline largest leave-one-worker-out connected component
            is_sorted (bool): if False, dataframe will be sorted by i (and t, if included). Returned dataframe will be sorted. Sorting may alter original dataframe if copy is set to False. Set is_sorted to True if dataframe is already sorted.
            copy (bool): if False, avoid copy
            first_loop (bool): if True, this is the first loop of the method

        Returns:
            (BipartiteLongBase): dataframe of largest leave-one-worker-out connected component
        '''
        # Sort and copy
        frame_init = self.sort_rows(is_sorted=is_sorted, copy=copy)

        for cc in sorted(cc_list, reverse=True, key=len):
            cc = np.array(cc)
            cc_j = cc[cc <= max_j]
            if frame_largest_cc is not None:
                # Can check if frame_cc is already smaller than frame_largest_cc before any computations
                skip = False
                if component_size_variable == 'firms':
                    # If looking at number of firms
                    try:
                        skip = (frame_largest_cc.comp_size >= len(cc_j))
                    except AttributeError:
                        frame_largest_cc.comp_size = frame_largest_cc.n_firms()
                        skip = (frame_largest_cc.comp_size >= len(cc_j))
                elif component_size_variable == 'firms_plus_movers':
                    # If looking at number of firms plus number of movers
                    try:
                        skip = (frame_largest_cc.comp_size >= len(cc))
                    except AttributeError:
                        frame_largest_cc.comp_size = frame_largest_cc.n_firms() + frame_largest_cc.loc[frame_largest_cc.loc[:, 'm'].to_numpy() > 0, :].n_workers()
                        skip = (frame_largest_cc.comp_size >= len(cc))
                if skip:
                    continue

            # Keep observations in connected components (NOTE: this does not require a copy)
            frame_cc = frame_init.keep_ids('j', cc_j, drop_returns_to_stays, is_sorted=True, copy=False)

            if frame_largest_cc is not None:
                # If frame_cc is already smaller than frame_largest_cc
                skip = bpd.util.compare_frames(frame_largest_cc, frame_cc, size_variable=component_size_variable, operator='geq', save_to_frame1=True, is_sorted=True)

                if skip:
                    continue

            # Remove firms with only 1 mover observation (can have 1 mover with multiple observations)
            frame_cc = frame_cc.min_moves_frame(2, drop_returns_to_stays, is_sorted=True, copy=False)

            if frame_largest_cc is not None:
                # If frame_cc is already smaller than frame_largest_cc
                skip = bpd.util.compare_frames(frame_largest_cc, frame_cc, size_variable=component_size_variable, operator='geq', save_to_frame1=True, is_sorted=True)

                if skip:
                    continue

            # Construct graph
            G2, max_j2 = frame_cc._construct_graph('leave_out_worker', is_sorted=True, copy=False)

            # Extract articulation workers
            articulation_ids = np.array(G2.articulation_points())
            articulation_workers = articulation_ids[articulation_ids > max_j2]
            articulation_workers -= (max_j2 + 1)

            if len(articulation_workers) > 0:
                # If new frame is not leave-one-worker-out connected, recompute connected components after dropping articulation workers (but note that articulation workers should be kept in the final dataframe) (NOTE: this does not require a copy)
                G2, max_j3 = frame_cc.drop_ids('i', articulation_workers, drop_returns_to_stays, is_sorted=True, reset_index=False, copy=False)._construct_graph('leave_out_worker', is_sorted=True, copy=False)
                cc_list_2 = G2.components()
                if len(cc_list_2) > 1:
                    # Recursion step (only necessary if dropping articulation workers disconnects the set of firms)
                    frame_cc = frame_cc._leave_out_worker(cc_list=cc_list_2, max_j=max_j3, component_size_variable=component_size_variable, drop_returns_to_stays=drop_returns_to_stays, frame_largest_cc=frame_largest_cc, is_sorted=True, copy=False, first_loop=False)

            if frame_largest_cc is None:
                # If in the first round
                replace = True
            elif frame_cc is None:
                # If the biconnected components have recursively been eliminated
                replace = False
            else:
                replace = bpd.util.compare_frames(frame_largest_cc, frame_cc, size_variable=component_size_variable, operator='lt', save_to_frame1=True, is_sorted=True)
            if replace:
                frame_largest_cc = frame_cc

        if first_loop:
            # Remove comp_size attribute before return
            try:
                del frame_largest_cc.comp_size
            except AttributeError:
                pass

        # Return largest leave-one-worker-out component
        return frame_largest_cc

    def _construct_firm_linkages(self, is_sorted=False, copy=True):
        '''
        Construct numpy array linking firms by movers, for use with connected components.

        Arguments:
            is_sorted (bool): not used for ._construct_firm_linkages()
            copy (bool): not used for ._construct_firm_linkages()

        Returns:
            (NumPy Array): firm linkages
            (int): maximum firm id
        '''
        move_rows = (self.loc[:, 'm'].to_numpy() > 0)
        i_col = self.loc[move_rows, 'i'].to_numpy()
        j_col = self.loc[move_rows, 'j'].to_numpy()
        j_next = np.roll(j_col, -1)
        i_match = (i_col == bpd.util.fast_shift(i_col, -1, fill_value=-2))
        j_col = j_col[i_match]
        j_next = j_next[i_match]
        linkages = np.stack([j_col, j_next], axis=1)
        max_j = np.max(linkages)

        return linkages, max_j

    def _construct_firm_double_linkages(self, is_sorted=False, copy=True):
        '''
        Construct numpy array linking firms by movers, for use with leave-one-firm-out components.

        Arguments:
            is_sorted (bool): not used for ._construct_firm_double_linkages()
            copy (bool): not used for ._construct_firm_double_linkages()

        Returns:
            (NumPy Array): firm linkages
            (int): maximum firm id
        '''
        move_rows = (self.loc[:, 'm'].to_numpy() > 0)
        i_col = self.loc[move_rows, 'i'].to_numpy()
        j_col = self.loc[move_rows, 'j'].to_numpy()
        i_next = bpd.util.fast_shift(i_col, -1, fill_value=-2)
        j_next = np.roll(j_col, -1)
        valid_next = (i_col == i_next)
        base_linkages = np.stack([j_col[valid_next], j_next[valid_next]], axis=1)
        i_next_2 = bpd.util.fast_shift(i_col, -2, fill_value=-2)
        j_next_2 = np.roll(j_col, -2)
        valid_next_2 = (i_col == i_next_2)
        secondary_linkages = np.stack([j_col[valid_next_2], j_next_2[valid_next_2]], axis=1)
        linkages = np.concatenate([base_linkages, secondary_linkages], axis=0)
        max_j = np.max(linkages)

        return linkages, max_j

    def _construct_firm_worker_linkages(self, is_sorted=False, copy=True):
        '''
        Construct numpy array linking firms to worker ids, for use with leave-one-observation-out components.

        Arguments:
            is_sorted (bool): if False, dataframe will be sorted by i in a groupby (but self will not be not sorted). Set to True if already sorted.
            copy (bool): not used for long format

        Returns:
            (NumPy Array): firm-worker linkages
            (int): maximum firm id
        '''
        worker_m = self.get_worker_m(is_sorted)
        i_col = self.loc[worker_m, 'i'].to_numpy()
        j_col = self.loc[worker_m, 'j'].to_numpy()
        max_j = np.max(j_col)
        linkages = np.stack([i_col + max_j + 1, j_col], axis=1)

        return linkages, max_j

    def keep_ids(self, id_col, keep_ids_list, drop_returns_to_stays=False, is_sorted=False, reset_index=True, copy=True):
        '''
        Only keep ids belonging to a given set of ids.

        Arguments:
            id_col (str): column of ids to consider ('i', 'j', or 'g')
            keep_ids_list (list): ids to keep
            drop_returns_to_stays (bool): used only if id_col is 'j' or 'g' and using BipartiteLongCollapsed format. If True, when recollapsing collapsed data, drop observations that need to be recollapsed instead of collapsing (this is for computational efficiency when re-collapsing data for leave-one-out connected components, where intermediate observations can be dropped, causing a worker who returns to a firm to become a stayer).
            is_sorted (bool): if False, dataframe may be sorted by i (and t, if included) if data is collapsed long format. Returned dataframe is not guaranteed to be sorted if original dataframe is not sorted. Sorting may alter original dataframe if copy is set to False. Set is_sorted to True if dataframe is already sorted.
            reset_index (bool): if True, reset index at end
            copy (bool): if False, avoid copy

        Returns:
            (BipartiteLongBase): dataframe with ids in the given set
        '''
        keep_ids_list = set(keep_ids_list)
        if len(keep_ids_list) == self.n_unique_ids(id_col):
            # If keeping everything
            if copy:
                return self.copy()
            return self

        frame = self.loc[self.loc[:, id_col].isin(keep_ids_list), :]

        if id_col in ['j', 'g']:
            if isinstance(frame, bpd.BipartiteLongCollapsed):
                # If BipartiteLongCollapsed
                frame = frame.recollapse(drop_returns_to_stays=drop_returns_to_stays, is_sorted=is_sorted, copy=copy)
                # We don't need to copy again
                copy = False

            # Recompute 'm' since it might change from dropping observations or from re-collapsing
            frame = frame.gen_m(force=True, copy=copy)
            # We don't need to copy again
            copy = False

        if copy:
            # Copy on subset
            frame = frame.copy()

        if reset_index:
            frame.reset_index(drop=True, inplace=True)

        return frame

    def drop_ids(self, id_col, drop_ids_list, drop_returns_to_stays=False, is_sorted=False, reset_index=True, copy=True):
        '''
        Drop ids belonging to a given set of ids.

        Arguments:
            id_col (str): column of ids to consider ('i', 'j', or 'g')
            drop_ids_list (list): ids to drop
            drop_returns_to_stays (bool): used only if id_col is 'j' or 'g' and using BipartiteLongCollapsed format. If True, when recollapsing collapsed data, drop observations that need to be recollapsed instead of collapsing (this is for computational efficiency when re-collapsing data for leave-one-out connected components, where intermediate observations can be dropped, causing a worker who returns to a firm to become a stayer).
            is_sorted (bool): if False, dataframe may be sorted by i (and t, if included) if data is collapsed long format. Returned dataframe is not guaranteed to be sorted if original dataframe is not sorted. Sorting may alter original dataframe if copy is set to False. Set is_sorted to True if dataframe is already sorted.
            reset_index (bool): if True, reset index at end
            copy (bool): if False, avoid copy

        Returns:
            (BipartiteLongBase): dataframe with ids outside the given set
        '''
        drop_ids_list = set(drop_ids_list)
        if len(drop_ids_list) == 0:
            # If nothing input
            if copy:
                return self.copy()
            return self

        frame = self.loc[~(self.loc[:, id_col].isin(drop_ids_list)), :]

        if id_col in ['j', 'g']:
            if isinstance(frame, bpd.BipartiteLongCollapsed):
                # If BipartiteLongCollapsed
                frame = frame.recollapse(drop_returns_to_stays=drop_returns_to_stays, is_sorted=is_sorted, copy=copy)
                # We don't need to copy again
                copy = False

            # Recompute 'm' since it might change from dropping observations or from re-collapsing
            frame = frame.gen_m(force=True, copy=copy)
            # We don't need to copy again
            copy = False

        if copy:
            # Copy on subset
            frame = frame.copy()

        if reset_index:
            frame.reset_index(drop=True, inplace=True)

        return frame

    def keep_rows(self, rows_list, drop_returns_to_stays=False, is_sorted=False, reset_index=True, copy=True):
        '''
        Only keep particular rows.

        Arguments:
            rows_list (list): rows to keep
            drop_returns_to_stays (bool): if True, when recollapsing collapsed data, drop observations that need to be recollapsed instead of collapsing (this is for computational efficiency when re-collapsing data for leave-one-out connected components, where intermediate observations can be dropped, causing a worker who returns to a firm to become a stayer)
            is_sorted (bool): if False, dataframe may be sorted by i (and t, if included) if data is collapsed long format. Returned dataframe is not guaranteed to be sorted if original dataframe is not sorted. Sorting may alter original dataframe if copy is set to False. Set is_sorted to True if dataframe is already sorted.
            reset_index (bool): if True, reset index at end
            copy (bool): if False, avoid copy

        Returns:
            (BipartiteLongBase): dataframe with given rows
        '''
        rows_list = set(rows_list)
        if len(rows_list) == len(self):
            # If keeping everything
            if copy:
                return self.copy()
            return self
        rows_list = sorted(list(rows_list))

        frame = self.iloc[rows_list]

        if isinstance(frame, bpd.BipartiteLongCollapsed):
            # If BipartiteLongCollapsed
            frame = frame.recollapse(drop_returns_to_stays=drop_returns_to_stays, is_sorted=is_sorted, copy=copy)
            # We don't need to copy again
            copy = False

        # Recompute 'm' since it might change from dropping observations or from re-collapsing
        frame = frame.gen_m(force=True, copy=copy)

        if reset_index:
            frame.reset_index(drop=True, inplace=True)

        return frame

    def min_obs_firms(self, threshold=2, is_sorted=False, copy=True):
        '''
        List firms with at least `threshold` many observations.

        Arguments:
            threshold (int): minimum number of observations required to keep a firm
            is_sorted (bool): not used for long format
            copy (bool): not used for long format

        Returns:
            (NumPy Array): firms with sufficiently many observations
        '''
        if threshold == 0:
            # If no threshold
            return self.unique_ids('j')

        n_obs = self.loc[:, 'j'].value_counts(sort=False)
        valid_firms = n_obs[n_obs.to_numpy() >= threshold].index.to_numpy()

        return valid_firms

    @bpd.bipartitebase._recollapse_loop(False)
    def min_obs_frame(self, threshold=2, drop_returns_to_stays=False, is_sorted=False, copy=True):
        '''
        Return dataframe of firms with at least `threshold` many observations.

        Arguments:
            threshold (int): minimum number of observations required to keep a firm
            drop_returns_to_stays (bool): if True, when recollapsing collapsed data, drop observations that need to be recollapsed instead of collapsing (this is for computational efficiency when re-collapsing data for leave-one-out connected components, where intermediate observations can be dropped, causing a worker who returns to a firm to become a stayer)
            is_sorted (bool): if False, dataframe may be sorted by i (and t, if included) if data is collapsed long format. Returned dataframe is not guaranteed to be sorted if original dataframe is not sorted. Sorting may alter original dataframe if copy is set to False. Set is_sorted to True if dataframe is already sorted.
            copy (bool): if False, avoid copy

        Returns:
            (BipartiteLongBase): dataframe of firms with sufficiently many observations
        '''
        if threshold == 0:
            # If no threshold
            if copy:
                return self.copy()
            return self

        frame = self.loc[self.groupby('j')['i'].transform('size').to_numpy() >= threshold, :]

        if isinstance(frame, bpd.BipartiteLongCollapsed):
            # If BipartiteLongCollapsed
            frame = frame.recollapse(drop_returns_to_stays=drop_returns_to_stays, is_sorted=is_sorted, copy=copy)
            # We don't need to copy again
            copy = False

        # Recompute 'm' since it might change from dropping observations or from re-collapsing
        frame = frame.gen_m(force=True, copy=copy)

        frame.reset_index(drop=True, inplace=True)

        return frame

    def min_workers_firms(self, threshold=15, is_sorted=False, copy=True):
        '''
        List firms with at least `threshold` many workers.

        Arguments:
            threshold (int): minimum number of workers required to keep a firm
            is_sorted (bool): not used for long format
            copy (bool): not used for long format

        Returns:
            (NumPy Array): list of firms with sufficiently many workers
        '''
        if threshold == 0:
            # If no threshold
            return self.unique_ids('j')

        n_workers = self.groupby('j')['i'].nunique()
        valid_firms = n_workers[n_workers.to_numpy() >= threshold].index.to_numpy()

        return valid_firms

    @bpd.bipartitebase._recollapse_loop(False)
    def min_workers_frame(self, threshold=15, drop_returns_to_stays=False, is_sorted=False, copy=True):
        '''
        Return dataframe of firms with at least `threshold` many workers.

        Arguments:
            threshold (int): minimum number of workers required to keep a firm
            drop_returns_to_stays (bool): if True, when recollapsing collapsed data, drop observations that need to be recollapsed instead of collapsing (this is for computational efficiency when re-collapsing data for leave-one-out connected components, where intermediate observations can be dropped, causing a worker who returns to a firm to become a stayer)
            is_sorted (bool): if False, dataframe may be sorted by i (and t, if included) if data is collapsed long format. Returned dataframe is not guaranteed to be sorted if original dataframe is not sorted. Sorting may alter original dataframe if copy is set to False. Set is_sorted to True if dataframe is already sorted.
            copy (bool): if False, avoid copy

        Returns:
            (BipartiteLongBase): dataframe of firms with sufficiently many workers
        '''
        if threshold == 0:
            # If no threshold
            if copy:
                return self.copy()
            return self

        frame = self.loc[self.groupby('j')['i'].transform('nunique').to_numpy() >= threshold, :]

        if isinstance(frame, bpd.BipartiteLongCollapsed):
            # If BipartiteLongCollapsed
            frame = frame.recollapse(drop_returns_to_stays=drop_returns_to_stays, is_sorted=is_sorted, copy=copy)
            # We don't need to copy again
            copy = False

        # Recompute 'm' since it might change from dropping observations or from re-collapsing
        frame = frame.gen_m(force=True, copy=copy)

        frame.reset_index(drop=True, inplace=True)

        return frame

    def min_moves_firms(self, threshold=2):
        '''
        List firms with at least `threshold` many moves. Note that a single mover can have multiple moves at the same firm.

        Arguments:
            threshold (int): minimum number of moves required to keep a firm

        Returns:
            (NumPy Array): firms with sufficiently many moves
        '''
        if threshold == 0:
            # If no threshold
            return self.unique_ids('j')

        return self.loc[self.loc[:, 'm'].to_numpy() > 0].min_obs_firms(threshold=threshold)

    @bpd.bipartitebase._recollapse_loop(True)
    def min_moves_frame(self, threshold=2, drop_returns_to_stays=False, is_sorted=False, reset_index=True, copy=True):
        '''
        Return dataframe where all firms have at least `threshold` many moves. Note that a single worker can have multiple moves at the same firm. This method employs loops, as dropping firms that don't meet the threshold may lower the number of moves at other firms.

        Arguments:
            threshold (int): minimum number of moves required to keep a firm
            drop_returns_to_stays (bool): if True, when recollapsing collapsed data, drop observations that need to be recollapsed instead of collapsing (this is for computational efficiency when re-collapsing data for leave-one-out connected components, where intermediate observations can be dropped, causing a worker who returns to a firm to become a stayer)
            is_sorted (bool): if False, dataframe may be sorted by i (and t, if included) if data is collapsed long format. Returned dataframe is not guaranteed to be sorted if original dataframe is not sorted. Sorting may alter original dataframe if copy is set to False. Set is_sorted to True if dataframe is already sorted.
            reset_index (bool): if True, reset index at end
            copy (bool): if False, avoid copy

        Returns:
            (BipartiteLongBase): dataframe of firms with sufficiently many moves
        '''
        if threshold == 0:
            # If no threshold
            if copy:
                return self.copy()
            return self

        valid_firms = self.min_moves_firms(threshold)

        return self.keep_ids('j', keep_ids_list=valid_firms, drop_returns_to_stays=drop_returns_to_stays, is_sorted=is_sorted, reset_index=reset_index, copy=copy)

    @bpd.bipartitebase._recollapse_loop(True)
    def min_movers_frame(self, threshold=15, drop_returns_to_stays=False, is_sorted=False, reset_index=True, copy=True):
        '''
        Return dataframe where all firms have at least `threshold` many movers. This method employs loops, as dropping firms that don't meet the threshold may lower the number of movers at other firms.

        Arguments:
            threshold (int): minimum number of movers required to keep a firm
            drop_returns_to_stays (bool): if True, when recollapsing collapsed data, drop observations that need to be recollapsed instead of collapsing (this is for computational efficiency when re-collapsing data for leave-one-out connected components, where intermediate observations can be dropped, causing a worker who returns to a firm to become a stayer)
            is_sorted (bool): if False, dataframe may be sorted by i (and t, if included) if data is collapsed long format. Returned dataframe is not guaranteed to be sorted if original dataframe is not sorted. Sorting may alter original dataframe if copy is set to False. Set is_sorted to True if dataframe is already sorted.
            reset_index (bool): if True, reset index at end
            copy (bool): if False, avoid copy

        Returns:
            (BipartiteLongBase): dataframe of firms with sufficiently many movers
        '''
        self.log('generating frame of firms with a minimum number of movers', level='info')

        if threshold == 0:
            # If no threshold
            if copy:
                return self.copy()
            return self

        valid_firms = self.min_movers_firms(threshold, is_sorted=False, copy=True)

        return self.keep_ids('j', keep_ids_list=valid_firms, drop_returns_to_stays=drop_returns_to_stays, is_sorted=is_sorted, reset_index=reset_index, copy=copy)

    def construct_artificial_time(self, time_per_worker=False, is_sorted=False, copy=True):
        '''
        Construct artificial time column(s) to enable conversion to (collapsed) event study format. Only adds column(s) if time column(s) not already included.

        Arguments:
            time_per_worker (bool): if True, set time independently for each worker (note that this is significantly more computationally costly)
            is_sorted (bool): set to True if dataframe is already sorted by i (this avoids a sort inside a groupby if time_per_worker=True, but this groupby will not sort the returned dataframe)
            copy (bool): if False, avoid copy

        Returns:
            (BipartiteLongBase): dataframe with artificial time column(s)
        '''
        if copy:
            frame = self.copy()
        else:
            frame = self

        if not frame._col_included('t'):
            if time_per_worker:
                # Reset time for each worker
                t = frame.groupby('i', sort=(not is_sorted)).cumcount()
            else:
                # Cumulative time over all workers
                t = np.arange(len(frame))
            for t_col in self.col_reference_dict['t']:
                frame.loc[:, t_col] = t

        # Sort columns
        frame = frame.sort_cols(copy=False)

        return frame
