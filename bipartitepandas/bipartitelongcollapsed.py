'''
Class for a bipartite network in collapsed long format.
'''
import numpy as np
import pandas as pd
import bipartitepandas as bpd

class BipartiteLongCollapsed(bpd.BipartiteLongBase):
    '''
    Class for bipartite networks of firms and workers in collapsed long form (i.e. employment spells are collapsed into a single observation). Inherits from BipartiteLongBase.

    Arguments:
        *args: arguments for BipartiteLongBase
        col_reference_dict (dict or None): clarify which columns are associated with a general column name, e.g. {'i': 'i', 'j': ['j1', 'j2']}; None is equivalent to {}
        col_collapse_dict (dict or None): how to collapse column (None indicates the column should be dropped), e.g. {'y': 'mean'}; None is equivalent to {}
        **kwargs: keyword arguments for BipartiteLongBase
    '''

    def __init__(self, *args, col_reference_dict=None, col_collapse_dict=None, **kwargs):
        # Update parameters to be lists/dictionaries instead of None (source: https://stackoverflow.com/a/54781084/17333120)
        if col_reference_dict is None:
            col_reference_dict = {}
        if col_collapse_dict is None:
            col_collapse_dict = {}
        col_reference_dict = bpd.util.update_dict({'t': ['t1', 't2']}, col_reference_dict)
        col_collapse_dict = bpd.util.update_dict({'m': None}, col_collapse_dict)
        # Initialize DataFrame
        super().__init__(*args, col_reference_dict=col_reference_dict, col_collapse_dict=col_collapse_dict, **kwargs)

        # self.log('BipartiteLongCollapsed object initialized', level='info')

    @property
    def _constructor(self):
        '''
        For inheritance from Pandas.

        Returns:
            (class): BipartiteLongCollapsed class
        '''
        return BipartiteLongCollapsed

    @property
    def _constructor_es(self):
        '''
        For .to_eventstudy(), tells BipartiteLongBase which event study format to use.

        Returns:
            (class): BipartiteEventStudyCollapsed class
        '''
        return bpd.BipartiteEventStudyCollapsed

    @property
    def _constructor_ees(self):
        '''
        For .to_extendedeventstudy(), tells BipartiteLongBase which event study format to use.

        Returns:
            (class): BipartiteExtendedEventStudyCollapsed class
        '''
        return bpd.BipartiteExtendedEventStudyCollapsed

    def get_worker_m(self, is_sorted=False):
        '''
        Get NumPy array indicating whether the worker associated with each observation is a mover.

        Arguments:
            is_sorted (bool): not used for collapsed long format

        Returns:
            (NumPy Array): indicates whether the worker associated with each observation is a mover
        '''
        return self.loc[:, 'm'].to_numpy() > 0

    def recollapse(self, drop_returns_to_stays=False, is_sorted=False, copy=True):
        '''
        Recollapse data by job spells (so each spell for a particular worker at a particular firm is one observation). This method is necessary in the case of biconnected data - it can occur that a worker works at firms A and B in the order A B A, but the biconnected components removes firm B. So the data is now A A, and needs to be recollapsed so this is marked as a stayer.

        Arguments:
            drop_returns_to_stays (bool): if True, when recollapsing collapsed data, drop observations that need to be recollapsed instead of collapsing (this is for computational efficiency when re-collapsing data for leave-one-out connected components, where intermediate observations can be dropped, causing a worker who returns to a firm to become a stayer)
            is_sorted (bool): if False, dataframe will be sorted by i (and t, if included). Returned dataframe will be sorted. Sorting may alter original dataframe if copy is set to False. Set is_sorted to True if dataframe is already sorted.
            copy (bool): if False, avoid copy

        Returns:
            (BipartiteLongCollapsed): recollapsed dataframe
        '''
        self.log('beginning recollapse', level='info')

        # Sort and copy
        frame = self.sort_rows(j_if_no_t=True, is_sorted=is_sorted, copy=copy)
        self.log('data sorted by i (and t, if included; and by j, if t not included)', level='info')

        # If no returns
        if frame.no_returns:
            return frame

        # Generate spell ids
        spell_ids = frame._get_spell_ids(is_sorted=True, copy=False)

        # Quickly check whether a recollapse is necessary
        if (len(frame) < 2) or (spell_ids[-1] == len(frame)):
            return frame

        # Indicator if must re-recollapse
        recursion = False

        if drop_returns_to_stays:
            ## Drop returns that turned into stays (i.e. only keep spells of size 1) ##
            # Aggregate at the spell level
            spells = frame.groupby(spell_ids, sort=False)

            data_spell = frame.loc[spells['i'].transform('size').to_numpy() == 1, :]
            if len(data_spell) < len(frame):
                # If recollapsed, it's possible another recollapse might be necessary
                recursion = True
        else:
            ### If recollapse necessary ###
            # Dictionary linking columns to how they should be aggregated
            agg_funcs = {}

            # Keep track of user-added columns
            user_added_cols = {}

            ## Correctly weight ##
            # If weight column exists
            weighted = frame._col_included('w')

            if weighted:
                ## Initial preparation if weighted ##
                w = frame.loc[:, 'w'].to_numpy()
                # Keep track of new, weighted columns
                weighted_cols = []
                for col in frame._included_cols():
                    if (col == 't') or (frame.col_collapse_dict[col] is None):
                        # If time, skip this column; if None, drop this column
                        pass
                    else:
                        # If not time column
                        aggfunc = frame.col_collapse_dict[col]
                        if aggfunc == 'mean':
                            # If column can be weighted, weight it
                            for subcol in bpd.util.to_list(frame.col_reference_dict[col]):
                                with bpd.util.ChainedAssignment():
                                    frame.loc[:, subcol + '_weighted'] = w * frame.loc[:, subcol].to_numpy()
                                weighted_cols.append(subcol + '_weighted')
                        elif aggfunc in ['var', 'std']:
                            # Variance and standard deviation can't be computed
                            for subcol in bpd.util.to_list(frame.col_reference_dict[col]):
                                frame.loc[:, subcol + '_weighted'] = np.nan
                                weighted_cols.append(subcol + '_weighted')

            ## Aggregate at the spell level ##
            spells = frame.groupby(spell_ids, sort=False)

            # First, prepare non-time columns for aggregation
            default_cols = frame.columns_req + frame.columns_opt
            for col in frame._included_cols():
                if (col == 't') or (frame.col_collapse_dict[col] is None):
                    # If time, skip this column; if None, drop this column
                    pass
                else:
                    # If not time column
                    aggfunc = frame.col_collapse_dict[col]
                    if weighted and (aggfunc == 'mean'):
                        # If column should be weighted
                        for subcol in bpd.util.to_list(frame.col_reference_dict[col]):
                            agg_funcs[subcol] = pd.NamedAgg(column=subcol + '_weighted', aggfunc='sum')
                    elif weighted and (aggfunc in ['var', 'std']):
                        # Variance and standard deviation can't be computed
                        for subcol in bpd.util.to_list(frame.col_reference_dict[col]):
                            agg_funcs[subcol] = pd.NamedAgg(column=subcol + '_weighted', aggfunc='first')
                    else:
                        # Unweighted column
                        for subcol in bpd.util.to_list(frame.col_reference_dict[col]):
                            agg_funcs[subcol] = pd.NamedAgg(column=subcol, aggfunc=aggfunc)
                    if col not in default_cols:
                        # User-added columns
                        user_added_cols[col] = frame.col_reference_dict[col]

            # Next, prepare the time column for aggregation
            if self._col_included('t'):
                agg_funcs['t1'] = pd.NamedAgg(column='t1', aggfunc='min')
                agg_funcs['t2'] = pd.NamedAgg(column='t2', aggfunc='max')

            # Next, prepare the weight column for aggregation
            if not weighted:
                agg_funcs['w'] = pd.NamedAgg(column='i', aggfunc='size')

            # Aggregate columns
            data_spell = spells.agg(**agg_funcs)

            # Finally, normalize weighted columns
            if weighted:
                w_sum = data_spell.loc[:, 'w'].to_numpy()
                for col in frame._included_cols():
                    if (col == 't') or (frame.col_collapse_dict[col] is None):
                        # If time, skip this column; if None, drop this column
                        pass
                    else:
                        # If not time column
                        aggfunc = frame.col_collapse_dict[col]
                        if aggfunc == 'mean':
                            # If column should be normalized
                            for subcol in bpd.util.to_list(frame.col_reference_dict[col]):
                                data_spell.loc[:, subcol] /= w_sum
                with bpd.util.ChainedAssignment():
                    # Drop added columns
                    frame = frame.drop(weighted_cols, axis=1, inplace=True)

        # Sort columns
        sorted_cols = bpd.util._sort_cols(data_spell.columns)
        data_spell = data_spell.reindex(sorted_cols, axis=1, copy=False)
        data_spell.reset_index(drop=True, inplace=True)

        self.log('data aggregated at the spell level', level='info')

        collapsed_frame = bpd.BipartiteLongCollapsed(data_spell, col_reference_dict=user_added_cols, log=frame._log_on_indicator)
        collapsed_frame._set_attributes(self, no_dict=False)

        for col, col_collapse in frame.col_collapse_dict.items():
            # Remove dropped columns from attribute dictionaries
            # NOTE: this must iterate over frame's dictionary, not collapsed_frame's dictionary, otherwise it raises an error
            if col_collapse is None:
                # If column should be dropped during collapse
                del collapsed_frame.col_dtype_dict[col]
                del collapsed_frame.col_collapse_dict[col]
                del collapsed_frame.col_long_es_dict[col]
                if col in collapsed_frame.columns_contig.keys():
                    # If column is categorical
                    del collapsed_frame.columns_contig[col]
                    if collapsed_frame.id_reference_dict:
                        # If linking contiguous ids to original ids
                        del collapsed_frame.id_reference_dict[col]

        # m can change from collapsing
        collapsed_frame = collapsed_frame.gen_m(force=True, copy=False)

        if recursion:
            self.log('must re-collapse again', level='info')
            return collapsed_frame.recollapse(drop_returns_to_stays=drop_returns_to_stays, is_sorted=True, copy=False)

        return collapsed_frame

    def uncollapse(self, drop_no_collapse_columns=True, is_sorted=False, copy=True):
        '''
        Return collapsed long data reformatted into long data, by assuming variables constant over spells.

        Arguments:
            drop_no_collapse_columns (bool): if True, columns marked by self.col_collapse_dict as None (i.e. they should be dropped) will not be dropped
            is_sorted (bool): if False, dataframe will be sorted by i (and t, if included). Returned dataframe will be sorted. Sorting may alter original dataframe if copy is set to False. Set is_sorted to True if dataframe is already sorted.
            copy (bool): if False, avoid copy

        Returns:
            (BipartiteLong): collapsed long data reformatted as long data
        '''
        # Sort and copy
        frame = self.sort_rows(is_sorted=is_sorted, copy=copy)

        # All included columns
        all_cols = frame._included_cols()

        # Keep track of user-added columns
        user_added_cols = {}
        default_cols = frame.columns_req + frame.columns_opt

        t = frame._col_included('t')
        if not t:
            # If no t column, no difference between long and collapsed long
            data_long = frame

            for col in all_cols:
                if (col not in default_cols) and ((frame.col_collapse_dict[col] is not None) or (not drop_no_collapse_columns)):
                    # User-added columns
                    user_added_cols[col] = frame.col_reference_dict[col]

        else:
            # Skip t1 and t2
            all_cols.remove('t')

            # Link columns to index, for use with frame.itertuples() (source: https://stackoverflow.com/a/36460020/17333120)
            col_to_idx = {k: v for v, k in enumerate(frame.columns)}

            # Dictionary of lists of each column's data
            long_dict = {'t': []}
            for col in all_cols:
                if (frame.col_collapse_dict[col] is not None) or (not drop_no_collapse_columns):
                    # Drop column if None and drop_no_collapse_columns is True
                    for subcol in bpd.util.to_list(frame.col_reference_dict[col]):
                        long_dict[subcol] = []
                    if col not in default_cols:
                        # User-added columns
                        user_added_cols[col] = frame.col_reference_dict[col]

            # Iterate over rows with multiple periods
            nt = frame.loc[:, 't2'].to_numpy() - frame.loc[:, 't1'].to_numpy() + 1

            for i, row in enumerate(frame.itertuples(index=False)):
                # Source: https://stackoverflow.com/a/41022840/17333120
                nt_i = nt[i]
                long_dict['t'].extend(range(row[col_to_idx['t1']], row[col_to_idx['t2']] + 1))
                for col in all_cols:
                    if (frame.col_collapse_dict[col] is not None) or (not drop_no_collapse_columns):
                        # Drop column if None and drop_no_collapse_columns is True
                        for subcol in bpd.util.to_list(frame.col_reference_dict[col]):
                            if frame.col_collapse_dict[col] == 'sum':
                                # Evenly split sum across periods
                                long_dict[subcol].extend([row[col_to_idx[subcol]] / nt_i] * nt_i)
                            else:
                                long_dict[subcol].extend([row[col_to_idx[subcol]]] * nt_i)
            del nt

            # Convert to Pandas dataframe
            data_long = pd.DataFrame(long_dict)

            # Sort columns
            sorted_cols = bpd.util._sort_cols(data_long.columns)
            data_long = data_long.reindex(sorted_cols, axis=1, copy=False)

        self.log('data uncollapsed to long format', level='info')

        long_frame = bpd.BipartiteLong(data_long, col_reference_dict=user_added_cols, log=frame._log_on_indicator)
        long_frame._set_attributes(frame, no_dict=True)

        if not drop_no_collapse_columns:
            # If shouldn't drop None columns, set None columns to have collapse of 'first' (this is because we don't want the column to drop during data cleaning)
            for col, col_collapse in frame.col_collapse_dict.items():
                if col_collapse is None:
                    long_frame.col_collapse_dict[col] = 'first'
        else:
            # If should drop None columns
            for col, col_collapse in frame.col_collapse_dict.items():
                # Remove dropped columns from attribute dictionaries
                if col_collapse is None:
                    # If column should be dropped during uncollapse
                    del long_frame.col_dtype_dict[col]
                    del long_frame.col_collapse_dict[col]
                    del long_frame.col_long_es_dict[col]
                    if col in long_frame.columns_contig.keys():
                        # If column is categorical
                        del long_frame.columns_contig[col]
                        if long_frame.id_reference_dict:
                            # If linking contiguous ids to original ids
                            del long_frame.id_reference_dict[col]

        if t:
            # Only recompute 'm' if data actually uncollapsed
            long_frame = long_frame.gen_m(force=True, copy=False)

        return long_frame

    def to_permutedeventstudy(self, order='sequential', move_to_worker=False, is_sorted=False, copy=True, rng=None):
        '''
        Return collapsed long form data reformatted into collapsed permuted event study data. In this method, permuting the data means combining each set of two observations drawn from a single worker into an event study observation (e.g. if a worker works at firms A, B, and C, this will create data with rows A-B; B-C; and A-C).

        Arguments:
            order (str): if 'sequential', each observation will be in sequential order; if 'income', order will be set based on the average income of the worker
            move_to_worker (bool): if True, each move is treated as a new worker
            is_sorted (bool): if False, dataframe will be sorted by i (and t, if included). Returned dataframe will be sorted. Sorting may alter original dataframe if copy is set to False. Set is_sorted to True if dataframe is already sorted.
            copy (bool): if False, avoid copy
            rng (np.random.Generator or None): NumPy random number generator; None is equivalent to np.random.default_rng(None)

        Returns:
            (Pandas DataFrame): permuted collapsed event study dataframe
        '''
        if rng is None:
            rng = np.random.default_rng(None)

        if not self.no_returns:
            raise ValueError("Cannot run method .to_permutedeventstudy() if there are returns in the data. When cleaning your data, please set the parameter 'drop_returns' to drop returns.")

        if order not in ['sequential', 'income']:
            raise ValueError(f"`order` must be either 'sequential' or 'income', but input specifies {order!r}.")

        # Sort and copy
        frame = self.sort_rows(is_sorted=is_sorted, copy=copy)

        # Split workers by movers and stayers
        worker_m = frame.get_worker_m(is_sorted=True)
        stayers = pd.DataFrame(frame.loc[~worker_m, :])
        movers = pd.DataFrame(frame.loc[worker_m, :])
        frame.log('workers split by movers and stayers', level='info')

        ## Figure out new indices for permutations ##
        if order == 'income':
            # Sort workers by average income
            with bpd.util.ChainedAssignment():
                if frame._col_included('w'):
                    movers.loc[:, 'weighted_y'] = movers.loc[:, 'w'].to_numpy() * movers.loc[:, 'y'].to_numpy()
                    movers['mean_y'] = movers.groupby('i')['weighted_y'].transform('sum') / movers.groupby('i')['w'].transform('sum')
                    movers.drop('weighted_y', axis=1, inplace=True)
                else:
                    movers['mean_y'] = movers.groupby('i')['y'].transform('mean')
                movers.sort_values('mean_y', inplace=True)
                movers.drop('mean_y', axis=1, inplace=True)

        # Initial data construction
        j = movers.loc[:, 'j'].to_numpy()
        idx = np.arange(len(movers))
        idx_1 = np.array([], dtype=int)
        idx_2 = np.array([], dtype=int)

        # Construct graph
        G, _ = frame._construct_graph(connectedness='leave_out_observation', is_sorted=True, copy=False)

        for j1 in range(self.n_firms()):
            ### Iterate over all firms ###
            # Find workers who worked at firm j1
            obs_in_j1 = (j == j1)
            movers.loc[:, 'obs_in_j1'] = obs_in_j1
            i_in_j1 = movers.groupby('i', sort=False)['obs_in_j1'].transform('max').to_numpy()
            # Take subset of data for workers who worked at firm j1
            movers_j1 = movers.loc[i_in_j1, :]
            j_j1 = j[i_in_j1]
            idx_j1 = idx[i_in_j1]
            # For each firm, find its neighboring firms
            j1_neighbors = G.neighborhood(j1, order=2, mindist=2)
            for j2 in j1_neighbors:
                ### Iterate over all neighbors ###
                if j2 > j1:
                    ## Account for symmetry by estimating only if j2 > j1 ##
                    # Find workers who worked at both firms j1 and j2
                    obs_in_j2 = (j_j1 == j2)
                    with bpd.util.ChainedAssignment():
                        movers_j1.loc[:, 'obs_in_j2'] = obs_in_j2
                    i_in_j2 = movers_j1.groupby('i', sort=False)['obs_in_j2'].transform('max').to_numpy()
                    # Take subsets of data for workers who worked at both firms j1 and j2
                    j_j12 = j_j1[i_in_j2]
                    idx_j12 = idx_j1[i_in_j2]
                    # Take subsets of data specifically for firms j1 and j2
                    is_j12 = (j_j12 == j1) | (j_j12 == j2)
                    j_j12 = j_j12[is_j12]
                    idx_j12 = idx_j12[is_j12]
                    # Split data for j1 and j2
                    idx_j11 = idx_j12[j_j12 == j1]
                    idx_j22 = idx_j12[j_j12 == j2]
                    # Split observations into entering/exiting groups
                    if order == 'sequential':
                        j_j12_first = j_j12[np.arange(len(j_j12)) % 2 == 0]
                        entering = (j_j12_first == j2)
                        exiting = (j_j12_first == j1)
                    elif order == 'income':
                        if len(idx_j11) % 2 == 0:
                            halfway = len(idx_j11) // 2
                        else:
                            halfway = len(idx_j11) // 2 + rng.binomial(n=1, p=0.5)
                        entering = (np.arange(len(idx_j11)) < halfway)
                        exiting = (np.arange(len(idx_j11)) >= halfway)
                    # Append new indices
                    idx_1 = np.append(idx_1, idx_j11[exiting])
                    idx_1 = np.append(idx_1, idx_j22[entering])
                    idx_2 = np.append(idx_2, idx_j22[exiting])
                    idx_2 = np.append(idx_2, idx_j11[entering])

        ## Add lagged values ##
        movers_permuted = pd.DataFrame()
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
                    ## Movers ##
                    # Useful for t1 and t2: t1 should go to t11 and t21; t2 should go to t12 and t22
                    col_1 = col + '1' + subcol_number
                    col_2 = col + '2' + subcol_number
                    # Lagged value
                    movers_permuted.loc[:, col_1] = movers.loc[:, subcol].to_numpy()[idx_1]
                    movers_permuted.loc[:, col_2] = movers.loc[:, subcol].to_numpy()[idx_2]

                    if subcol != 'i':
                        ## Stayers (no lags) ##
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
        movers_permuted = movers_permuted.loc[movers_permuted.loc[:, 'i1'].to_numpy() == movers_permuted.loc[:, 'i2'].to_numpy(), :]

        # Set 'm' for movers
        movers_permuted.loc[:, 'm'] = 1

        # Correct datatypes (shifting adds nans which converts all columns into float, correct columns that should be int)
        for col in all_cols:
            if ((frame.col_dtype_dict[col] == 'int') or (col in frame.columns_contig.keys())) and frame.col_long_es_dict[col]:
                for subcol in bpd.util.to_list(frame.col_reference_dict[col]):
                    # Get column number, e.g. j1 will give 1
                    subcol_number = subcol.strip(col)
                    shifted_col = col + '1' + subcol_number
                    movers_permuted.loc[:, shifted_col] = movers_permuted.loc[:, shifted_col].astype(int, copy=False)

        # Correct i
        movers_permuted.drop('i2', axis=1, inplace=True)
        movers_permuted.rename({'i1': 'i'}, axis=1, inplace=True)

        # Keep only relevant columns
        stayers = stayers.reindex(keep_cols, axis=1, copy=False)
        movers_permuted = movers_permuted.reindex(keep_cols, axis=1, copy=False)
        frame.log('columns updated', level='info')

        # Merge stayers and movers (NOTE: this converts the data into a Pandas DataFrame)
        data_es = pd.concat([stayers, movers_permuted], ignore_index=True) # .reset_index(drop=True)

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
        if not is_sorted:
            raise NotImplementedError('._get_spell_ids() requires `is_sorted` == True, but it is set to False.')

        if self.no_returns:
            # If no returns, then each observation is guaranteed to be a new spell
            self.log('preparing to compute spell ids', level='info')

            spell_ids = np.arange(len(self))
            self.log('spell ids generated', level='info')

            return spell_ids
        return super()._get_spell_ids(is_sorted=is_sorted, copy=copy)

    def _drop_i_t_duplicates(self, how='max', is_sorted=False, copy=True):
        '''
        Keep only the highest paying job for i-t (worker-year) duplicates.

        Arguments:
            how (str or function): if 'max', keep max paying job; otherwise, take `how` over duplicate worker-firm-year observations, then take the highest paying worker-firm observation. `how` can take any input valid for a Pandas transform.
            is_sorted (bool): if False, dataframe will be sorted by i (and t, if included). Returned dataframe will be sorted. Sorting may alter original dataframe if copy is set to False. Set is_sorted to True if dataframe is already sorted.
            copy (bool): if False, avoid copy

        Returns:
            (BipartiteLongCollapsed): dataframe that keeps only the highest paying job for i-t (worker-year) duplicates. If no t column(s), returns frame with no changes
        '''
        if copy:
            frame = self.copy()
        else:
            frame = self

        if frame._col_included('t'):
            ## Convert to long ##
            # Keep track of columns that aren't supposed to convert to long, but we allow to convert because this is during data cleaning
            no_collapse_cols = [col for col, col_collapse in frame.col_collapse_dict.items() if col_collapse is None]

            frame = frame.uncollapse(drop_no_collapse_columns=False, is_sorted=is_sorted, copy=False)

            frame = frame._drop_i_t_duplicates(how, is_sorted=True, copy=False)

            # Return to collapsed long
            frame = frame.collapse(is_sorted=True, copy=False)

            # Update col_collapse_dict for columns that aren't supposed to convert to long
            for col in no_collapse_cols:
                frame.col_collapse_dict[col] = None

            # Data now has unique i-t observations
            frame.i_t_unique = True
        else:
            frame.i_t_unique = None

        return frame

    def _get_articulation_observations(self, G, max_j, is_sorted=False):
        '''
        Compute articulation observations for self, by checking whether self is leave-one-observation-out connected when dropping selected observations one at a time.

        Arguments:
            G (igraph Graph): graph linking firms by movers
            max_j (int): maximum j
            is_sorted (bool): not used for collapsed long format

        Returns:
            (NumPy Array): indices of articulation observations
        '''
        # Find bridges (recall i is adjusted to be greater than j, which is why we reverse the order) (source for manual method: https://igraph.discourse.group/t/function-to-find-edges-which-are-bridges-in-r-igraph/154/2)
        # NOTE: use built-in for single observations, manual for spells/matches
        bridges = [G.es[bridge].tuple for bridge in G.bridges()] # [a for a in G.biconnected_components() if len(a) == 2]
        bridges_workers = set([bridge[1] - (max_j + 1) for bridge in bridges])
        bridges_firms = set([bridge[0] for bridge in bridges])

        # Find articulation observations (source for alternative: https://stackoverflow.com/a/55893561/17333120)
        articulation_rows = self.index[self.loc[:, 'i'].isin(bridges_workers) & self.loc[:, 'j'].isin(bridges_firms)].to_numpy() # self.index[self.set_index(['i', 'j']).index.isin(bridges)].to_numpy()

        # # Get possible articulation observations
        # # possible_articulation_obs = self.loc[pd.Series(map(tuple, self.loc[:, ['i', 'j']].to_numpy())).reindex_like(self, copy=False).isin(bridges), ['i', 'j', 'm']] # FIXME this doesn't work
        # possible_articulation_obs = self.loc[self.loc[:, 'i'].isin(bridges_workers) & self.loc[:, 'j'].isin(bridges_firms), ['i', 'j', 'm']]

        # # Find articulation observations - an observation is an articulation observation if the firm-worker pair has only a single observation
        # if self.no_returns:
        #     # If no returns, every row is guaranteed to be an articulation observation
        #     articulation_rows = possible_articulation_obs.index.to_numpy()
        # else:
        #     # If returns, then returns will have multiple worker-firm pairs, meaning they are not articulation observations
        #     articulation_rows = possible_articulation_obs.index.to_numpy()[possible_articulation_obs.groupby(['i', 'j'], sort=True)['m'].transform('size').to_numpy() == 1]

        return articulation_rows

    def _get_articulation_spells(self, G, max_j, is_sorted=False):
        '''
        Compute articulation spells for self, by checking whether self is leave-one-spell-out connected when dropping selected spells one at a time.

        Arguments:
            G (igraph Graph): graph linking firms by movers
            max_j (int): maximum j
            is_sorted (bool): not used for collapsed long format

        Returns:
            (NumPy Array): indices of articulation spells
        '''
        # Since spells are equivalent to observations with collapsed data, articulation spells are articulation observations
        return self._get_articulation_observations(G=G, max_j=max_j, is_sorted=is_sorted)
