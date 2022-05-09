'''
Class for a bipartite network in collapsed long format.
'''
from numpy import arange
import pandas as pd
import bipartitepandas as bpd

class BipartiteLongCollapsed(bpd.BipartiteLongBase):
    '''
    Class for bipartite networks of firms and workers in collapsed long form (i.e. employment spells are collapsed into a single observation). Inherits from BipartiteLongBase.

    Arguments:
        *args: arguments for BipartiteLongBase
        col_reference_dict (dict or None): clarify which columns are associated with a general column name, e.g. {'wid': 'wid', 'j': ['j1', 'j2']}; None is equivalent to {}
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

        ## Aggregate at the spell level
        spell = frame.groupby(spell_ids, sort=False)

        # Indicator if must re-recollapse
        recursion = False

        if drop_returns_to_stays:
            data_spell = frame.loc[spell['i'].transform('size').to_numpy() == 1, :]
            if len(data_spell) < len(frame):
                # If recollapsed, it's possible another recollapse might be necessary
                recursion = True
        else:
            # Dictionary linking columns to how they should be aggregated
            agg_funcs = {}

            # Keep track of user-added columns
            user_added_cols = {}

            # First, prepare non-time columns for aggregation
            default_cols = frame.columns_req + frame.columns_opt
            for col in frame._included_cols():
                if col != 't':
                    # Skip time column
                    for subcol in bpd.util.to_list(frame.col_reference_dict[col]):
                        if frame.col_collapse_dict[col] is not None:
                            # If column should be collapsed
                            agg_funcs[subcol] = pd.NamedAgg(column=subcol, aggfunc=frame.col_collapse_dict[col])
                    if col not in default_cols:
                        # User-added columns
                        user_added_cols[col] = frame.col_reference_dict[col]

            # Next, prepare the time column for aggregation
            if self._col_included('t'):
                agg_funcs['t1'] = pd.NamedAgg(column='t1', aggfunc='min')
                agg_funcs['t2'] = pd.NamedAgg(column='t2', aggfunc='max')

            # Next, prepare the weight column for aggregation
            if 'w' not in agg_funcs.keys():
                agg_funcs['w'] = pd.NamedAgg(column='i', aggfunc='size')

            # Finally, aggregate
            data_spell = spell.agg(**agg_funcs)

        # Sort columns
        sorted_cols = bpd.util._sort_cols(data_spell.columns)
        data_spell = data_spell.reindex(sorted_cols, axis=1, copy=False)
        data_spell.reset_index(drop=True, inplace=True)

        self.log('data aggregated at the spell level', level='info')

        collapsed_frame = bpd.BipartiteLongCollapsed(data_spell, col_reference_dict=user_added_cols, log=frame._log_on_indicator)
        collapsed_frame._set_attributes(self, no_dict=False)

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

            spell_ids = arange(len(self))
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
            ## Convert to long
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
        # Find bridges (recall i is adjusted to be greater than j, which is why we reverse the order) (source: https://igraph.discourse.group/t/function-to-find-edges-which-are-bridges-in-r-igraph/154/2)
        bridges = [tuple(sorted(a, reverse=True)) for a in G.biconnected_components() if len(a) == 2]
        bridges_workers = set([bridge[0] - (max_j + 1) for bridge in bridges])
        bridges_firms = set([bridge[1] for bridge in bridges])

        # Get possible articulation observations
        # possible_articulation_obs = self.loc[pd.Series(map(tuple, self.loc[:, ['i', 'j']].to_numpy())).reindex_like(self, copy=False).isin(bridges), ['i', 'j', 'm']] # FIXME this doesn't work
        possible_articulation_obs = self.loc[self.loc[:, 'i'].isin(bridges_workers) & self.loc[:, 'j'].isin(bridges_firms), ['i', 'j', 'm']]

        # Find articulation observations - an observation is an articulation observation if the firm-worker pair has only a single observation
        if self.no_returns:
            # If no returns, every row is guaranteed to be an articulation observation
            articulation_rows = possible_articulation_obs.index.to_numpy()
        else:
            # If returns, then returns will have multiple worker-firm pairs, meaning they are not articulation observations
            articulation_rows = possible_articulation_obs.index.to_numpy()[possible_articulation_obs.groupby(['i', 'j'], sort=True)['m'].transform('size').to_numpy() == 1]

        return articulation_rows

    def _get_articulation_spells(self, G, max_j, is_sorted=False, copy=True):
        '''
        Compute articulation spells for self, by checking whether self is leave-one-spell-out connected when dropping selected spells one at a time.

        Arguments:
            G (igraph Graph): graph linking firms by movers
            max_j (int): maximum j
            is_sorted (bool): not used for collapsed long format
            copy (bool): not used for collapsed long format

        Returns:
            (NumPy Array): indices of articulation spells
        '''
        # Since spells are equivalent to observations with collapsed data, articulation spells are articulation observations
        return self._get_articulation_observations(G=G, max_j=max_j, is_sorted=is_sorted)
