'''
Class for a bipartite network in long format.
'''
import numpy as np
import pandas as pd
import bipartitepandas as bpd
import warnings

class BipartiteLong(bpd.BipartiteLongBase):
    '''
    Class for bipartite networks of firms and workers in long form. Inherits from BipartiteLongBase.

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
        col_reference_dict = bpd.util.update_dict({'t': 't'}, col_reference_dict)
        col_collapse_dict = bpd.util.update_dict({'m': 'sum'}, col_collapse_dict)
        # Initialize DataFrame
        super().__init__(*args, col_reference_dict=col_reference_dict, col_collapse_dict=col_collapse_dict, **kwargs)

        # self.log('BipartiteLong object initialized', level='info')

    @property
    def _constructor(self):
        '''
        For inheritance from Pandas.

        Returns:
            (class): BipartiteLong class
        '''
        return BipartiteLong

    @property
    def _constructor_es(self):
        '''
        For .to_eventstudy(), tells BipartiteLongBase which event study format to use.

        Returns:
            (class): BipartiteEventStudy class
        '''
        return bpd.BipartiteEventStudy

    @property
    def _constructor_ees(self):
        '''
        For .to_extendedeventstudy(), tells BipartiteLongBase which event study format to use.

        Returns:
            (class): BipartiteExtendedEventStudy class
        '''
        return bpd.BipartiteExtendedEventStudy

    def get_worker_m(self, is_sorted=False):
        '''
        Get NumPy array indicating whether the worker associated with each observation is a mover.

        Arguments:
            is_sorted (bool): if False, dataframe will be sorted by i in a groupby (but self will not be not sorted). Set is_sorted to True if dataframe is already sorted.

        Returns:
            (NumPy Array): indicates whether the worker associated with each observation is a mover
        '''
        return self.groupby('i', sort=(not is_sorted))['m'].transform('max').to_numpy() > 0

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
        else:
            params = params.copy()

        self.log('beginning BipartiteLong data cleaning', level='info')

        connectedness = params['connectedness']
        collapse_connectedness = params['collapse_at_connectedness_measure']
        # If will have to collapse the data after cleaning
        collapse = (collapse_connectedness and connectedness in ['leave_out_spell', 'leave_out_match'])
        if collapse:
            params['connectedness'] = None

        ## Initial cleaning ##
        frame = super().clean(params)

        if collapse:
            ## Collapse then compute largest connected set ##
            # Update parameters
            level_dict = {
                'leave_out_spell': 'spell',
                'leave_out_match': 'match'
            }
            # NOTE: leave-out-observation is equivalent to leave-out-(spell/match) if the data is collapsed at the (spell/match) level, but the code is faster
            params['connectedness'] = 'leave_out_observation'
            params['drop_returns'] = False
            params['is_sorted'] = True
            params['copy'] = False
            params['force'] = False

            # Collapse
            frame = frame.collapse(level=level_dict[connectedness], is_sorted=True, copy=False)

            # Clean
            frame = frame.clean(params)

        self.log('BipartiteLongBase data cleaning complete', level='info')

        return frame

    def collapse(self, level='spell', is_sorted=False, copy=True):
        '''
        Collapse long data at the worker-firm spell/match level (so each spell/match for a particular worker at a particular firm becomes one observation).

        Arguments:
            level (str): if 'spell', collapse at the worker-firm spell level; if 'match', collapse at the worker-firm match level ('spell' and 'match' will differ if a worker leaves then returns to a firm)
            is_sorted (bool): if False, dataframe will be sorted by i (and t, if included). Returned dataframe will be sorted. Sorting may alter original dataframe if copy is set to False. Set is_sorted to True if dataframe is already sorted.
            copy (bool): if False, avoid copy

        Returns:
            (BipartiteLongCollapsed): collapsed long data generated by collapsing long data at the worker-firm spell level
        '''
        self.log(f'beginning collapse, level={level!r}', level='info')

        # Sort and copy
        frame = self.sort_rows(j_if_no_t=True, is_sorted=is_sorted, copy=copy)
        self.log('data sorted by i (and t, if included; and by j, if t not included)', level='info')

        # Generate group ids
        if level == 'spell':
            group_ids = frame._get_spell_ids(is_sorted=True, copy=False)
        elif level == 'match':
            group_ids = frame.groupby(['i', 'j'], sort=(not frame.no_returns)).ngroup().to_numpy()
        else:
            raise ValueError(f"`level` must be one of 'spell' or 'match', but input specifies invalid input {level!r}.")

        if (len(frame) < 2) or (group_ids[-1] == len(frame)):
            ## Quickly check whether a collapse is necessary ##
            # Keep track of user-added columns
            user_added_cols = {}
            default_cols = frame.columns_req + frame.columns_opt
            for col in frame._included_cols():
                if (col != 't') and (col not in default_cols) and (frame.col_collapse_dict[col] is not None):
                    # User-added columns that should remain
                    user_added_cols[col] = frame.col_reference_dict[col]

            # Update time column
            if frame._col_included('t'):
                with bpd.util.ChainedAssignment():
                    frame.loc[:, 't1'] = frame.loc[:, 't']
                    frame.loc[:, 't2'] = frame.loc[:, 't']
                frame.drop('t', axis=1, inplace=True, allow_optional=True)

            # Assign data_spell
            data_spell = frame
        else:
            ### If collapse necessary ###
            # Dictionary linking columns to how they should be aggregated
            agg_funcs = {}

            # Keep track of user-added columns
            user_added_cols = {}

            ## Correctly weight ##
            # If weight column exists
            weighted = frame._col_included('w')

            if weighted:
                ## Initial preparation if weighted ##
                any_var = False
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
                        if aggfunc in ['mean', 'var', 'std']:
                            # If column can be weighted, weight it
                            for subcol in bpd.util.to_list(frame.col_reference_dict[col]):
                                with bpd.util.ChainedAssignment():
                                    frame.loc[:, subcol + '_weighted'] = w * frame.loc[:, subcol].to_numpy()
                                weighted_cols.append(subcol + '_weighted')
                            if aggfunc in ['var', 'std']:
                                any_var = True

            ## Aggregate at the spell level ##
            spells = frame.groupby(group_ids, sort=not ((level == 'match') and (not frame.no_returns)))

            if weighted and any_var:
                ## Variance computation if weighted ##
                w_sum = spells['w'].transform('sum').to_numpy()
                for col in frame._included_cols():
                    if (col == 't') or (frame.col_collapse_dict[col] is None):
                        # If time, skip this column; if None, drop this column
                        pass
                    else:
                        # If not time column
                        aggfunc = frame.col_collapse_dict[col]
                        if aggfunc in ['var', 'std']:
                            # If computing weighted variance
                            for subcol in bpd.util.to_list(frame.col_reference_dict[col]):
                                subcol_mean = spells[subcol + '_weighted'].transform('sum').to_numpy() / w_sum
                                with bpd.util.ChainedAssignment():
                                    frame.loc[:, subcol + '_weighted'] = w * ((frame.loc[:, subcol].to_numpy() - subcol_mean) ** 2)

                ## Re-aggregate at the spell level ##
                spells = frame.groupby(group_ids, sort=not ((level == 'match') and (not frame.no_returns)))

            # First, prepare non-time columns for aggregation
            default_cols = frame.columns_req + frame.columns_opt
            for col in frame._included_cols():
                if (col == 't') or (frame.col_collapse_dict[col] is None):
                    # If time, skip this column; if None, drop this column
                    pass
                else:
                    # If not time column
                    aggfunc = frame.col_collapse_dict[col]
                    if weighted and (aggfunc in ['mean', 'var', 'std']):
                        # If column should be weighted
                        for subcol in bpd.util.to_list(frame.col_reference_dict[col]):
                            agg_funcs[subcol] = pd.NamedAgg(column=subcol + '_weighted', aggfunc='sum')
                    else:
                        # Unweighted column
                        for subcol in bpd.util.to_list(frame.col_reference_dict[col]):
                            agg_funcs[subcol] = pd.NamedAgg(column=subcol, aggfunc=aggfunc)
                    if col not in default_cols:
                        # User-added columns
                        user_added_cols[col] = frame.col_reference_dict[col]

            # Next, prepare the time column for aggregation
            if self._col_included('t'):
                agg_funcs['t1'] = pd.NamedAgg(column='t', aggfunc='min')
                agg_funcs['t2'] = pd.NamedAgg(column='t', aggfunc='max')

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
                        if aggfunc in ['mean', 'var', 'std']:
                            # If column should be normalized
                            for subcol in bpd.util.to_list(frame.col_reference_dict[col]):
                                data_spell.loc[:, subcol] /= w_sum
                                if aggfunc == 'std':
                                    # Take square root of variance
                                    with bpd.util.ChainedAssignment():
                                        data_spell.loc[:, subcol] = np.sqrt(data_spell.loc[:, subcol].to_numpy())
                with bpd.util.ChainedAssignment():
                    # Drop added columns
                    frame = frame.drop(weighted_cols, axis=1, inplace=True)

            # Sort columns
            sorted_cols = bpd.util._sort_cols(data_spell.columns)
            data_spell = data_spell.reindex(sorted_cols, axis=1, copy=False)
            data_spell.reset_index(drop=True, inplace=True)

        self.log(f'data aggregated at the {level!r} level', level='info')

        collapsed_frame = bpd.BipartiteLongCollapsed(data_spell, col_reference_dict=user_added_cols, log=frame._log_on_indicator)
        collapsed_frame._set_attributes(frame, no_dict=True)

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

        if not ((len(frame) < 2) or (group_ids[-1] == len(frame))):
            # If data actually collapsed, m can change from collapsing
            collapsed_frame = collapsed_frame.gen_m(force=True, copy=False)

        # If no time column, or level == 'match', then returns will be collapsed
        if not collapsed_frame._col_included('t') or (level == 'match'):
            collapsed_frame.no_returns = True

        return collapsed_frame

    def _drop_i_t_duplicates(self, how='max', is_sorted=False, copy=True):
        '''
        Keep only the highest paying job for i-t (worker-year) duplicates.

        Arguments:
            how (str or function): if 'max', keep max paying job; otherwise, take `how` over duplicate worker-firm-year observations, then take the highest paying worker-firm observation. `how` can take any input valid for a Pandas transform.
            is_sorted (bool): if False, dataframe will be sorted by i (and t, if included). Returned dataframe will be sorted. Sorting may alter original dataframe if copy is set to False. Set is_sorted to True if dataframe is already sorted.
            copy (bool): if False, avoid copy

        Returns:
            (BipartiteLong): dataframe that keeps only the highest paying job for i-t (worker-year) duplicates. If no t column(s), returns frame with no changes
        '''
        if copy:
            frame = self.copy()
        else:
            frame = self

        if frame._col_included('t'):
            # Check whether any observations are dropped
            prev_len = len(frame)

            frame = frame.sort_rows(is_sorted=is_sorted, copy=False)
            # Temporarily disable warnings
            warnings.filterwarnings('ignore')
            if how not in ['max', max]:
                # Group by worker id, time, and firm id, and take `how` of compensation
                frame.loc[:, 'y'] = frame.groupby(['i', 't', 'j'])['y'].transform(how)
            # Take max over duplicates
            # Source: https://stackoverflow.com/questions/23394476/keep-other-columns-when-doing-groupby
            frame = frame.loc[frame.loc[:, 'y'].to_numpy() == frame.groupby(['i', 't'], sort=False)['y'].transform('max').to_numpy(), :].groupby(['i', 't'], as_index=False, sort=False).first()
            # Restore warnings
            warnings.filterwarnings('default')

            # Data now has unique i-t observations
            frame.i_t_unique = True

            # If observations dropped, recompute 'm'
            if prev_len != len(frame):
                frame = frame.gen_m(force=True, copy=False)
        else:
            frame.i_t_unique = None

        return frame

    def _get_articulation_observations(self, G, max_j, is_sorted=False):
        '''
        Compute articulation observations for self, by checking whether self is leave-one-observation-out connected when dropping selected observations one at a time.

        Arguments:
            G (igraph Graph): graph linking firms by movers
            max_j (int): maximum j
            is_sorted (bool): if False, dataframe will be sorted by i and j in a groupby (but self will not be sorted). Set is_sorted to True if dataframe is already sorted by i.

        Returns:
            (NumPy Array): indices of articulation observations
        '''
        if not is_sorted:
            raise NotImplementedError('._get_articulation_observations() requires `is_sorted` == True, but it is set to False.')

        # Find bridges (recall i is adjusted to be greater than j, which is why we reverse the order) (source for manual method: https://igraph.discourse.group/t/function-to-find-edges-which-are-bridges-in-r-igraph/154/2)
        # NOTE: use built-in for single observations, manual for spells/matches
        bridges = [G.es[bridge].tuple for bridge in G.bridges()] # [a for a in G.biconnected_components() if len(a) == 2]
        bridges_workers = set([bridge[1] - (max_j + 1) for bridge in bridges])
        bridges_firms = set([bridge[0] for bridge in bridges])
        # bridges = [(bridge[1] - (max_j + 1), bridge[0]) for bridge in bridges]

        # Find articulation observations (source for alternative: https://stackoverflow.com/a/55893561/17333120)
        articulation_rows = self.index[self.loc[:, 'i'].isin(bridges_workers) & self.loc[:, 'j'].isin(bridges_firms)].to_numpy() # self.index[self.set_index(['i', 'j']).index.isin(bridges)].to_numpy()

        # # Get possible articulation observations
        # # possible_articulation_obs = self.loc[pd.Series(map(tuple, self.loc[:, ['i', 'j']].to_numpy())).reindex_like(self, copy=False).isin(bridges), ['i', 'j', 'm']] # FIXME this doesn't work
        # possible_articulation_obs = self.loc[self.loc[:, 'i'].isin(bridges_workers) & self.loc[:, 'j'].isin(bridges_firms), ['i', 'j', 'm']]

        # # Find articulation observations - an observation is an articulation observation if the firm-worker pair has only a single observation
        # articulation_rows = possible_articulation_obs.index.to_numpy()[possible_articulation_obs.groupby(['i', 'j'], sort=(not (is_sorted and self.no_returns)))['m'].transform('size').to_numpy() == 1]

        return articulation_rows

    def _get_articulation_spells(self, G, max_j, is_sorted=False):
        '''
        Compute articulation spells for self, by checking whether self is leave-one-spell-out connected when dropping selected spells one at a time. (Note: spell ids are generated for this method and are generated on sorted data, so it is recommended to sort your data using .sort_rows() prior to calling this method, then run the method with is_sorted=True.)

        Arguments:
            G (igraph Graph): graph linking firms by movers
            max_j (int): maximum j
            is_sorted (bool): if False, dataframe will be sorted by i and j in a groupby (but self will not be sorted). Set is_sorted to True if dataframe is already sorted by i.

        Returns:
            (NumPy Array): indices of articulation spells
        '''
        if not is_sorted:
            raise NotImplementedError('._get_articulation_spells() requires `is_sorted` == True, but it is set to False.')

        # Find bridges (recall i is adjusted to be greater than j, which is why we reverse the order) (source for manual method: https://igraph.discourse.group/t/function-to-find-edges-which-are-bridges-in-r-igraph/154/2)
        # NOTE: use built-in for single observations, manual for spells/matches
        bridges = [a for a in G.biconnected_components() if len(a) == 2] # [G.es[bridge].tuple for bridge in G.bridges()]
        bridges_workers = set([bridge[1] - (max_j + 1) for bridge in bridges])
        bridges_firms = set([bridge[0] for bridge in bridges])

        # Find articulation spells - a spell is an articulation spell if the particular firm-worker pair has only a single spell
        if self.no_returns:
            # If no returns, every spell is guaranteed to be an articulation spell
            articulation_rows = self.index[self.loc[:, 'i'].isin(bridges_workers) & self.loc[:, 'j'].isin(bridges_firms)].to_numpy()
        else:
            ## If returns, then returns will have multiple worker-firm pairs, meaning they are not articulation spells ##
            # Get possible articulation spells
            # possible_articulation_spells = self.loc[pd.Series(map(tuple, self.loc[:, ['i', 'j']].to_numpy())).reindex_like(self, copy=False).isin(bridges), ['i', 'j', 'm']] # FIXME this doesn't work
            possible_articulation_rows = (self.loc[:, 'i'].isin(bridges_workers) & self.loc[:, 'j'].isin(bridges_firms)).to_numpy()
            possible_articulation_spells = self.loc[possible_articulation_rows, ['i', 'j', 'm']]
            spell_ids = self._get_spell_ids(is_sorted=True, copy=False)
            possible_articulation_spells['spell_id'] = spell_ids[possible_articulation_rows]
            articulation_rows = possible_articulation_spells.index.to_numpy()[possible_articulation_spells.groupby(['i', 'j'], sort=True)['spell_id'].transform('nunique').to_numpy() == 1]
            possible_articulation_spells.drop('spell_id', axis=1, inplace=True)

        return articulation_rows

    def fill_missing_periods(self, fill_dict=None, is_sorted=False, copy=True):
        '''
        Return Pandas dataframe of long format data with missing periods filled in as unemployed. By default j is filled in as - 1, and y and m are filled in as pd.NA, but these values can be specified.

        Arguments:
            fill_dict (dict or None): dictionary linking general column to value to fill in for missing rows. None is equivalent to {}. Set value to 'prev' to set to previous value that appeared in the dataframe (cannot use 'next' because this method iterates forward over the dataframe). Can set value for any column except i. Any column not listed will default to pd.NA, except 'j' will always default to -1 unless overridden.
            is_sorted (bool): if False, dataframe will be sorted by i and t. Returned dataframe will be sorted. Sorting may alter original dataframe if copy is set to False. Set is_sorted to True if dataframe is already sorted.
            copy (bool): if False, avoid copy

        Returns:
            (Pandas DataFrame): dataframe with missing periods filled in as unemployed
        '''
        if not self._col_included('t'):
            # Check whether t column included
            raise NotImplementedError('.fill_missing_periods() requires a time column, but dataframe does not include one.')

        if fill_dict is None:
            fill_dict = {}

        if 'i' in fill_dict.keys():
            raise NotImplementedError("Cannot set the value for 'i' in fill_dict.")

        # Update fill_dict
        fill_dict = bpd.util.update_dict({'j': -1}, fill_dict)

        # Sort, copy, reset index, and convert to Pandas dataframe
        frame = self.sort_rows(is_sorted=is_sorted, copy=copy)
        frame.reset_index(drop=True, inplace=True)
        frame = pd.DataFrame(frame, copy=False)

        # All included columns (minus i and t)
        all_cols = self._included_cols()
        all_cols.remove('i')
        all_cols.remove('t')

        # Fill in fill_dict for all columns
        for col in all_cols:
            if col not in fill_dict.keys():
                fill_dict[col] = pd.NA

        # Link columns to index, for use with frame.itertuples() (source: https://stackoverflow.com/a/36460020/17333120)
        col_to_idx = {k: v for v, k in enumerate(frame.columns)}

        # Dictionary of lists of each column's data
        filled_dict = {subcol: [] for col in all_cols for subcol in bpd.util.to_list(self.col_reference_dict[col])}
        filled_dict['i'] = []
        filled_dict['t'] = []

        # Find missing periods
        i_col = frame.loc[:, 'i'].to_numpy()
        t_col = frame.loc[:, 't'].to_numpy()
        i_next = bpd.util.fast_shift(i_col, -1, fill_value=-2)
        t_next = np.roll(t_col, -1)
        missing_periods = (i_col == i_next) & (t_col + 1 != t_next)

        if np.sum(missing_periods) > 0:
            t_col = t_col[missing_periods]
            t_next = t_next[missing_periods]
            del i_col, i_next

            # Compute how many periods are missing between each set of consecutive observations
            nt = (t_next - t_col - 1)

            # If have data to fill in
            for i, row in enumerate(frame.loc[missing_periods, :].itertuples(index=False)):
                # Source: https://stackoverflow.com/a/41022840/17333120
                nt_i = nt[i]
                # Start with i and t columns
                filled_dict['i'].extend([row[col_to_idx['i']]] * nt_i)
                filled_dict['t'].extend(np.arange(t_col[i] + 1, t_next[i]))
                for col in all_cols:
                    # Then do all other columns
                    for subcol in bpd.util.to_list(self.col_reference_dict[col]):
                        if isinstance(fill_dict[col], str) and (fill_dict[col] == 'prev'):
                            # Need isinstance check, because pd.NA == 'prev' raises an error
                            filled_dict[subcol].extend([row[col_to_idx[subcol]]] * nt_i)
                        else:
                            filled_dict[subcol].extend([fill_dict[col]] * nt_i)
            del nt

            # Convert to Pandas dataframe
            data_filled = pd.DataFrame(filled_dict)

            # Concatenate original data with filled data
            frame = pd.concat([frame, data_filled])

            # Sort data by i, t
            frame.sort_values(['i', 't'], inplace=True)
            frame.reset_index(drop=True, inplace=True)

            # Sort columns
            sorted_cols = bpd.util._sort_cols(frame.columns)
            frame = frame.reindex(sorted_cols, axis=1, copy=False)

        return frame
