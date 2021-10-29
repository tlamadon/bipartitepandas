'''
Class for a bipartite network in collapsed long form
'''
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

    def recollapse(self, copy=True):
        '''
        Recollapse data by job spells (so each spell for a particular worker at a particular firm is one observation). This method is necessary in the case of biconnected data - it can occur that a worker works at firms A and B in the order A B A, but the biconnected components removes firm B. So the data is now A A, and needs to be recollapsed so this is marked as a stayer.

        Arguments:
            copy (bool): if False, avoid copy

        Returns:
            frame (BipartiteBase): BipartiteBase with ids in the given set
        '''
        frame = pd.DataFrame(self, copy=copy)

        # Sort data by i and t
        frame = frame.sort_values(['i', 't1'])

        # Add w
        if 'w' not in frame.columns:
            frame['w'] = 1

        # Introduce lagged i and j
        frame['i_l1'] = frame['i'].shift(periods=1)
        frame['j_l1'] = frame['j'].shift(periods=1)

        # Generate spell ids
        new_spell = (frame['j'] != frame['j_l1']) | (frame['i'] != frame['i_l1']) # Allow for i != i_l1 to ensure that consecutive workers at the same firm get counted as different spells
        frame['spell_id'] = new_spell.cumsum()

        # Aggregate at the spell level
        spell = frame.groupby(['spell_id'])
        # First, aggregate required columns
        data_spell = spell.agg(
            i=pd.NamedAgg(column='i', aggfunc='first'),
            j=pd.NamedAgg(column='j', aggfunc='first'),
            y=pd.NamedAgg(column='y', aggfunc='mean'),
            t1=pd.NamedAgg(column='t1', aggfunc='min'),
            t2=pd.NamedAgg(column='t2', aggfunc='max'),
            w=pd.NamedAgg(column='w', aggfunc='sum')
        )
        # Next, aggregate optional columns
        all_cols = self._included_cols()
        for col in all_cols:
            if col in self.columns_opt:
                if self.col_dtype_dict[col] == 'int':
                    for subcol in bpd.to_list(self.reference_dict[col]):
                        data_spell[subcol] = spell[subcol].first()
                if self.col_dtype_dict[col] == 'float':
                    for subcol in bpd.to_list(self.reference_dict[col]):
                        data_spell[subcol] = spell[subcol].mean()

        collapsed_data = data_spell.reset_index(drop=True)

        # Sort columns
        sorted_cols = sorted(collapsed_data.columns, key=bpd.col_order)
        collapsed_data = collapsed_data[sorted_cols]

        collapsed_frame = bpd.BipartiteLongCollapsed(collapsed_data)
        collapsed_frame._set_attributes(self, no_dict=False)

        return collapsed_frame

    def uncollapse(self):
        '''
        Return collapsed long data reformatted into long data, by assuming variables constant over spells.

        Returns:
            long_frame (BipartiteLong): collapsed long data reformatted as BipartiteLong data
        '''
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
            for t in range(int(row['t1']), int(row['t2']) + 1):
                long_dict['t'].append(t)
                for col in all_cols: # Add variables other than period
                    long_dict[col].append(row[col])

        # Convert to Pandas dataframe
        data_long = pd.DataFrame(long_dict)
        # Correct datatypes
        data_long['t'] = data_long['t'].astype(int)
        data_long = data_long.astype({col: self.col_dtype_dict[col] for col in all_cols})

        # Sort columns
        sorted_cols = sorted(data_long.columns, key=bpd.col_order)
        data_long = data_long[sorted_cols]

        self.logger.info('data uncollapsed to long format')

        long_frame = bpd.BipartiteLong(data_long)
        long_frame._set_attributes(self, no_dict=True)

        return long_frame
