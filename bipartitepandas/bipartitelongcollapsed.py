'''
Class for a bipartite network in collapsed long form
'''
import numpy as np
import pandas as pd
import bipartitepandas as bpd

class BipartiteLongCollapsed(bpd.BipartiteLongBase):
    '''
    Class for bipartite networks of firms and workers in collapsed long form (i.e. employment spells are collapsed into a single observation). Inherits from BipartiteLongBase.

    Arguments:
        *args: arguments for BipartiteLongBase
        col_reference_dict (dict): clarify which columns are associated with a general column name, e.g. {'wid': 'wid', 'j': ['j1', 'j2']}
        col_collapse_dict (dict): how to collapse column (None indicates the column should be dropped), e.g. {'y': 'mean'}
        **kwargs: keyword arguments for BipartiteLongBase
    '''

    def __init__(self, *args, col_reference_dict={}, col_collapse_dict={}, **kwargs):
        col_reference_dict = bpd.update_dict({'t': ['t1', 't2']}, col_reference_dict)
        col_collapse_dict = bpd.update_dict({'m': 'first'}, col_collapse_dict)
        # Initialize DataFrame
        super().__init__(*args, col_reference_dict=col_reference_dict, col_collapse_dict=col_collapse_dict, **kwargs)

        # self.log('BipartiteLongCollapsed object initialized', level='info')

    @property
    def _constructor(self):
        '''
        For inheritance from Pandas.
        '''
        return BipartiteLongCollapsed

    @property
    def _constructor_es(self):
        '''
        For get_es(), tells BipartiteLongBase which event study format to use.

        Returns:
            (BipartiteEventStudyCollapsed): class
        '''
        return bpd.BipartiteEventStudyCollapsed

    def get_worker_m(self, is_sorted=False):
        '''
        Get NumPy array indicating whether the worker associated with each observation is a mover.

        Arguments:
            is_sorted (bool): used for long format, does nothing for collapsed long

        Returns:
            (NumPy Array): indicates whether the worker associated with each observation is a mover
        '''
        return self.loc[:, 'm'].to_numpy() > 0

    def recollapse(self, drop_returns_to_stays=False, is_sorted=False, copy=True):
        '''
        Recollapse data by job spells (so each spell for a particular worker at a particular firm is one observation). This method is necessary in the case of biconnected data - it can occur that a worker works at firms A and B in the order A B A, but the biconnected components removes firm B. So the data is now A A, and needs to be recollapsed so this is marked as a stayer.

        Arguments:
            drop_returns_to_stays (bool): if True, when recollapsing collapsed data, drop observations that need to be recollapsed instead of collapsing (this is for computational efficiency when re-collapsing data for leave-one-out connected components, where intermediate observations can be dropped, causing a worker who returns to a firm to become a stayer)
            is_sorted (bool): if False, dataframe will be sorted by i (and t, if included; and by j, if t not included). Set to True if already sorted.
            copy (bool): if False, avoid copy

        Returns:
            frame (BipartiteBase): BipartiteBase with ids in the given set
        '''
        self.log('beginning recollapse', level='info')

        # Sort data by i (and t, if included; and by j, if t not included)
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
                    for subcol in bpd.to_list(frame.col_reference_dict[col]):
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
        sorted_cols = sorted(data_spell.columns, key=bpd.col_order)
        data_spell = data_spell.reindex(sorted_cols, axis=1, copy=False)
        data_spell.reset_index(drop=True, inplace=True)

        self.log('data aggregated at the spell level', level='info')

        collapsed_frame = bpd.BipartiteLongCollapsed(data_spell, col_reference_dict=user_added_cols, log=frame._log_on_indicator)
        collapsed_frame._set_attributes(self, no_dict=False)

        if recursion:
            self.log('must re-collapse again', level='info')
            return collapsed_frame.recollapse(drop_returns_to_stays=drop_returns_to_stays, is_sorted=True, copy=False)

        return collapsed_frame

    def uncollapse(self, is_sorted=False, copy=True):
        '''
        Return collapsed long data reformatted into long data, by assuming variables constant over spells.

        Arguments:
            is_sorted (bool): if False, dataframe will be sorted by i (and t, if included). Set to True if already sorted.
            copy (bool): if False, avoid copy

        Returns:
            long_frame (BipartiteLong): collapsed long data reformatted as BipartiteLong data
        '''
        # Sort data by i (and t, if included)
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
                if col not in default_cols:
                    # User-added columns
                    user_added_cols[col] = frame.col_reference_dict[col]

        else:
            # Skip t1 and t2
            all_cols.remove('t')
            # Dictionary of lists of each column's data
            long_dict = {'t': []}
            for col in all_cols:
                if frame.col_collapse_dict[col] is not None:
                    # Drop column if None
                    for subcol in frame.col_reference_dict[col]:
                        long_dict[subcol] = []
                    if col not in default_cols:
                        # User-added columns
                        user_added_cols[col] = frame.col_reference_dict[col]

            # Iterate over rows with multiple periods
            nt = frame.loc[:, 't2'].to_numpy() - frame.loc[:, 't1'].to_numpy() + 1
            # Link columns to index, for use with frame.itertuples()
            col_to_idx = {}
            col_idx = list(frame.columns).index
            col_to_idx['t1'] = col_idx('t1')
            col_to_idx['t2'] = col_idx('t2')
            for col in all_cols:
                for subcol in frame.col_reference_dict[col]:
                    col_to_idx[subcol] = col_idx(subcol)
            for i, row in enumerate(frame.itertuples(index=False)):
                # Source: https://stackoverflow.com/a/41022840/17333120
                nt_i = nt[i]
                long_dict['t'].extend(np.arange(row[col_to_idx['t1']], row[col_to_idx['t2']] + 1))
                for col in all_cols:
                    if frame.col_collapse_dict[col] is not None:
                        # Drop column if None
                        for subcol in frame.col_reference_dict[col]:
                            if frame.col_collapse_dict[col] == 'sum':
                                # Evenly split sum across periods
                                long_dict[subcol].extend([row[col_to_idx[subcol]] / nt_i] * nt_i)
                            else:
                                long_dict[subcol].extend([row[col_to_idx[subcol]]] * nt_i)
            del nt

            # Convert to Pandas dataframe
            data_long = pd.DataFrame(long_dict)
            # Correct datatypes
            data_long.loc[:, 't'] = data_long.loc[:, 't'].astype(int, copy=False)
            for col in all_cols:
                if frame.col_collapse_dict[col] is not None:
                    # Drop column if None
                    for subcol in frame.col_reference_dict[col]:
                        if frame.col_dtype_dict[col] == 'int':
                            # If should be int
                            data_long.loc[:, subcol] = data_long.loc[:, subcol].astype(int, copy=False)

            # Sort columns
            sorted_cols = sorted(data_long.columns, key=bpd.col_order)
            data_long = data_long.reindex(sorted_cols, axis=1, copy=False)

        self.log('data uncollapsed to long format', level='info')

        long_frame = bpd.BipartiteLong(data_long, col_reference_dict=user_added_cols, log=frame._log_on_indicator)
        long_frame._set_attributes(frame, no_dict=True)
        if t:
            # Only recompute 'm' if data actually uncollapsed
            long_frame = long_frame.gen_m(force=True, copy=False)

        return long_frame

    def _drop_i_t_duplicates(self, how='max', is_sorted=False, copy=True):
        '''
        Keep only the highest paying job for i-t (worker-year) duplicates.

        Arguments:
            how (str or function): if 'max', keep max paying job; otherwise, take `how` over duplicate worker-firm-year observations, then take the highest paying worker-firm observation. `how` can take any input valid for a Pandas transform.
            is_sorted (bool): if False, dataframe will be sorted by i (and t, if included). Set to True if already sorted.
            copy (bool): if False, avoid copy

        Returns:
            frame (BipartiteLongCollapsed): dataframe that keeps only the highest paying job for i-t (worker-year) duplicates. If no t column(s), returns frame with no changes
        '''
        if copy:
            frame = self.copy()
        else:
            frame = self

        if frame._col_included('t'):
            # Convert to long
            frame = frame.uncollapse(is_sorted=is_sorted, copy=False)

            frame = frame._drop_i_t_duplicates(how, is_sorted=True, copy=False)

            # Return to collapsed long
            frame = frame.get_collapsed_long(is_sorted=True, copy=False)

        # Data now has unique i-t observations
        frame.i_t_unique = True

        return frame

    def _get_articulation_obs(self, G, max_j, is_sorted=False):
        '''
        Compute articulation observations for self, by checking whether self is leave-one-observation-out connected when dropping selected observations one at a time.

        Arguments:
            G (igraph Graph): graph linking firms by movers
            max_j (int): maximum j
            is_sorted (bool): used for long format, does nothing for collapsed long

        Returns:
            (NumPy Array): indices of articulation observations
        '''
        # Find bridges (recall i is adjusted to be greater than j, which is why we reverse the order) (source: https://igraph.discourse.group/t/function-to-find-edges-which-are-bridges-in-r-igraph/154/2)
        bridges = [tuple(sorted(a, reverse=True)) for a in G.biconnected_components() if len(a) == 2]
        bridges_workers = set([bridge[0] - (max_j + 1) for bridge in bridges])
        bridges_firms = set([bridge[1] for bridge in bridges])

        # Get possible articulation observations
        possible_articulation_obs = self.loc[self.loc[:, 'i'].isin(bridges_workers) & self.loc[:, 'j'].isin(bridges_firms), ['i', 'j', 'm']]
        # possible_articulation_obs = self.loc[pd.Series(map(tuple, self.loc[:, ['i', 'j']].to_numpy())).reindex_like(self, copy=False).isin(bridges), ['i', 'j', 'm']] # FIXME this doesn't work

        # Find articulation observations - an observation is an articulation observation if the firm-worker pair has only a single observation
        if self.no_returns:
            # If no returns, every row is guaranteed to be an articulation observation
            articulation_rows = possible_articulation_obs.index.to_numpy()
        else:
            # If returns, then returns will have multiple worker-firm pairs, meaning they are not articulation observations
            articulation_rows = possible_articulation_obs.index.to_numpy()[possible_articulation_obs.groupby(['i', 'j'], sort=True)['m'].transform('size').to_numpy() == 1]

        return articulation_rows
