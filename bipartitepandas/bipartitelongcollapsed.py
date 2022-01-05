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
        *args: arguments for Pandas DataFrame
        col_dict (dict): make data columns readable (requires: i (worker id), j (firm id), y (compensation), t1 (first period in spell), t2 (last period in spell); optionally include: w (weight), g (firm cluster), m (0 if stayer, 1 if mover)). Keep None if column names already correct
        include_id_reference_dict (bool): if True, create dictionary of Pandas dataframes linking original id values to contiguous id values
        **kwargs: keyword arguments for Pandas DataFrame
    '''

    def __init__(self, *args, col_dict=None, include_id_reference_dict=False, **kwargs):
        columns_opt = ['w']
        reference_dict = {'t': ['t1', 't2'], 'w': 'w'}
        col_dtype_dict = {'w': 'float'}
        # Initialize DataFrame
        super().__init__(*args, columns_opt=columns_opt, reference_dict=reference_dict, col_dtype_dict=col_dtype_dict, col_dict=col_dict, include_id_reference_dict=include_id_reference_dict, **kwargs)

        # self.logger.info('BipartiteLongCollapsed object initialized')

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

    def recollapse(self, drop_multiples=False, is_sorted=False, copy=True):
        '''
        Recollapse data by job spells (so each spell for a particular worker at a particular firm is one observation). This method is necessary in the case of biconnected data - it can occur that a worker works at firms A and B in the order A B A, but the biconnected components removes firm B. So the data is now A A, and needs to be recollapsed so this is marked as a stayer.

        Arguments:
            drop_multiples (bool): if True, rather than collapsing over spells, drop any spells with multiple observations (this is for computational efficiency when computing biconnected components)
            is_sorted (bool): if False, dataframe will be sorted by i (and t, if included). Set to True if already sorted.
            copy (bool): if False, avoid copy

        Returns:
            frame (BipartiteBase): BipartiteBase with ids in the given set
        '''
        frame = pd.DataFrame(self, copy=copy)

        if not is_sorted:
            if self._col_included('t'):
                # Sort data by i and t
                frame.sort_values(['i', 't1'], inplace=True)
            else:
                # Sort data by i
                frame.sort_values(['i'], inplace=True)

        # Add w
        if 'w' not in frame.columns:
            frame.loc[:, 'w'] = 1

        # Introduce lagged i and j
        i_col = frame.loc[:, 'i'].to_numpy()
        j_col = frame.loc[:, 'j'].to_numpy()
        i_prev = np.roll(i_col, 1)
        j_prev = np.roll(j_col, 1)

        # Generate spell ids
        new_spell = (j_col != j_prev) | (i_col != i_prev) # Allow for i != i_prev to ensure that consecutive workers at the same firm get counted as different spells
        del i_col, j_col, i_prev, j_prev
        spell_id = new_spell.cumsum()

        # Quickly check whether a recollapse is necessary
        if (len(frame) < 2) or (spell_id[-1] == len(frame)):
            if copy:
                return self.copy()
            return self

        # Aggregate at the spell level
        spell = frame.groupby(spell_id)

        if drop_multiples:
            data_spell = frame.loc[spell['i'].transform('size').to_numpy() == 1, :]
            data_spell.reset_index(drop=True, inplace=True)
        else:
            # First, aggregate required columns
            if self._col_included('t'):
                data_spell = spell.agg(
                    i=pd.NamedAgg(column='i', aggfunc='first'),
                    j=pd.NamedAgg(column='j', aggfunc='first'),
                    y=pd.NamedAgg(column='y', aggfunc='mean'),
                    t1=pd.NamedAgg(column='t1', aggfunc='min'),
                    t2=pd.NamedAgg(column='t2', aggfunc='max'),
                    w=pd.NamedAgg(column='w', aggfunc='sum')
                )
            else:
                data_spell = spell.agg(
                    i=pd.NamedAgg(column='i', aggfunc='first'),
                    j=pd.NamedAgg(column='j', aggfunc='first'),
                    y=pd.NamedAgg(column='y', aggfunc='mean'),
                    w=pd.NamedAgg(column='w', aggfunc='sum')
                )
            # Next, aggregate optional columns
            all_cols = self._included_cols()
            for col in all_cols:
                if col in self.columns_opt:
                    if self.col_dtype_dict[col] == 'int':
                        for subcol in bpd.to_list(self.reference_dict[col]):
                            data_spell.loc[:, subcol] = spell[subcol].first()
                    if self.col_dtype_dict[col] == 'float':
                        for subcol in bpd.to_list(self.reference_dict[col]):
                            data_spell.loc[:, subcol] = spell[subcol].mean()

            data_spell.reset_index(drop=True, inplace=True)

        # Sort columns
        sorted_cols = sorted(data_spell.columns, key=bpd.col_order)
        data_spell = data_spell.reindex(sorted_cols, axis=1, copy=False)

        collapsed_frame = bpd.BipartiteLongCollapsed(data_spell)
        collapsed_frame._set_attributes(self, no_dict=False)

        return collapsed_frame

    def uncollapse(self, is_sorted=False):
        '''
        Return collapsed long data reformatted into long data, by assuming variables constant over spells.

        Arguments:
            is_sorted (bool): if False, dataframe will be sorted by i (and t, if included). Set to True if already sorted.
            copy (bool): if False, avoid copy

        Returns:
            long_frame (BipartiteLong): collapsed long data reformatted as BipartiteLong data
        '''
        # Sort data by i and t
        self.sort_rows(is_sorted=is_sorted, copy=False)

        all_cols = self._included_cols(flat=True)
        # Skip t1 and t2
        all_cols.remove('t1')
        all_cols.remove('t2')
        long_dict = {'t': []} # Dictionary of lists of each column's data
        for col in all_cols:
            long_dict[col] = []

        # Iterate over all data
        for i in range(len(self)):
            row = self.iloc[i]
            for t in range(int(row.loc['t1']), int(row.loc['t2']) + 1):
                long_dict['t'].append(t)
                for col in all_cols: # Add variables other than period
                    long_dict[col].append(row[col])

        # Convert to Pandas dataframe
        data_long = pd.DataFrame(long_dict)
        # Correct datatypes
        data_long.loc[:, 't'] = data_long.loc[:, 't'].astype(int, copy=False)
        data_long = data_long.astype({col: self.col_dtype_dict[col] for col in all_cols}, copy=False)

        # Sort columns
        sorted_cols = sorted(data_long.columns, key=bpd.col_order)
        data_long = data_long.reindex(sorted_cols, axis=1, copy=False)

        self.logger.info('data uncollapsed to long format')

        long_frame = bpd.BipartiteLong(data_long)
        long_frame._set_attributes(self, no_dict=True)
        long_frame = long_frame.gen_m(force=True, copy=False)

        return long_frame

    def _drop_i_t_duplicates(self, how='max', is_sorted=False, copy=True):
        '''
        Keep only the highest paying job for i-t (worker-year) duplicates.

        Arguments:
            how (str): if 'max', keep max paying job; otherwise, take `how` over duplicate worker-firm-year observations, then take the highest paying worker-firm observation. `how` can take any option valid for a Pandas transform. Note that if multiple time and/or firm columns are included (as in event study format), then duplicates are cleaned in order of earlier time columns to later time columns, and earlier firm ids to later firm ids
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
            frame = frame.uncollapse(is_sorted=is_sorted)

            frame = bpd.BipartiteBase._drop_i_t_duplicates(frame, how, is_sorted=True, copy=False)

            # Return to collapsed long
            frame = frame.get_collapsed_long(is_sorted=True, copy=False)

        # Data now has unique i-t observations
        frame.i_t_unique = True

        return frame
