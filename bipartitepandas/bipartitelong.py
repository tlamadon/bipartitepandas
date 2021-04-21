'''
Class for a bipartite network in long form
'''
import pandas as pd
import bipartitepandas as bpd

class BipartiteLong(bpd.BipartiteLongBase):
    '''
    Class for bipartite networks of firms and workers in long form. Inherits from BipartiteLongBase.

    Arguments:
        *args: arguments for Pandas DataFrame
        col_dict (dict or None): make data columns readable (requires: i (worker id), j (firm id), y (compensation), t (period); optionally include: g (firm cluster), m (0 if stayer, 1 if mover)). Keep None if column names already correct
        include_id_reference_dict (bool): if True, create dictionary of Pandas dataframes linking original id values to contiguous id values
        **kwargs: keyword arguments for Pandas DataFrame
    '''

    def __init__(self, *args, col_dict=None, include_id_reference_dict=False, **kwargs):
        # Initialize DataFrame
        reference_dict = {'t': 't'}
        super().__init__(*args, reference_dict=reference_dict, col_dict=col_dict, include_id_reference_dict=include_id_reference_dict, **kwargs)

        # self.logger.info('BipartiteLong object initialized')

    @property
    def _constructor(self):
        '''
        For inheritance from Pandas.
        '''
        return BipartiteLong

    @property
    def _constructor_es(self):
        '''
        For get_es(), tells BipartiteLongBase which event study format to use.

        Returns:
            (BipartiteEventStudy): class
        '''
        return bpd.BipartiteEventStudy

    def get_collapsed_long(self):
        '''
        Collapse long data by job spells (so each spell for a particular worker at a particular firm is one observation).

        Returns:
            collapsed_frame (BipartiteLongCollapsed): BipartiteLongCollapsed object generated from long data collapsed by job spells
        '''
        # Generate m column (the function checks if it already exists)
        self.gen_m()

        # Copy data
        data = pd.DataFrame(self, copy=True)
        # Sort data by i and t
        data = data.sort_values(['i', 't'])
        self.logger.info('copied data sorted by i and t')

        # Introduce lagged i and j
        data['i_l1'] = data['i'].shift(periods=1)
        data['j_l1'] = data['j'].shift(periods=1)
        self.logger.info('lagged i and j introduced')

        # Generate spell ids
        # Source: https://stackoverflow.com/questions/59778744/pandas-grouping-and-aggregating-consecutive-rows-with-same-value-in-column
        new_spell = (data['j'] != data['j_l1']) | (data['i'] != data['i_l1']) # Allow for i != i_l1 to ensure that consecutive workers at the same firm get counted as different spells
        data['spell_id'] = new_spell.cumsum()
        self.logger.info('spell ids generated')

        # Aggregate at the spell level
        spell = data.groupby(['spell_id'])
        # First, aggregate required columns
        data_spell = spell.agg(
            i=pd.NamedAgg(column='i', aggfunc='first'),
            j=pd.NamedAgg(column='j', aggfunc='first'),
            y=pd.NamedAgg(column='y', aggfunc='mean'),
            t1=pd.NamedAgg(column='t', aggfunc='min'),
            t2=pd.NamedAgg(column='t', aggfunc='max'),
            w=pd.NamedAgg(column='i', aggfunc='size')
        )
        # Next, aggregate optional columns
        all_cols = self.included_cols()
        for col in all_cols:
            if col in self.columns_opt:
                if self.col_dtype_dict[col] == 'int':
                    for subcol in bpd.to_list(self.reference_dict[col]):
                        data_spell[subcol] = spell[subcol].first()
                if self.col_dtype_dict[col] == 'float':
                    for subcol in bpd.to_list(self.reference_dict[col]):
                        data_spell[subcol] = spell[subcol].mean()

        # # Classify movers and stayers
        # if not self.col_included('m'):
        #     spell_count = data_spell.groupby(['i']).transform('count')['j'] # Choice of j arbitrary
        #     data_spell['m'] = (spell_count > 1).astype(int)
        collapsed_data = data_spell.reset_index(drop=True)

        # Sort columns
        sorted_cols = sorted(collapsed_data.columns, key=bpd.col_order)
        collapsed_data = collapsed_data[sorted_cols]

        self.logger.info('data aggregated at the spell level')

        collapsed_frame = bpd.BipartiteLongCollapsed(collapsed_data)
        collapsed_frame.set_attributes(self, no_dict=True)

        return collapsed_frame

    def fill_periods(self, fill_j=-1, fill_y=pd.NA):
        '''
        Return Pandas dataframe of long form data with missing periods filled in as unemployed. By default j is filled in as - 1 and y is filled in as pd.NA, but these values can be specified.

        Arguments:
            fill_j (value): value to fill in for missing j
            fill_y (value): value to fill in for missing y

        Returns:
            fill_frame (Pandas DataFrame): Pandas DataFrame with missing periods filled in as unemployed
        '''
        import numpy as np
        m = self.col_included('m') # Check whether m column included
        fill_frame = pd.DataFrame(self, copy=True).sort_values(['i', 't']).reset_index(drop=True) # Sort by i, t
        fill_frame['i_l1'] = fill_frame['i'].shift(periods=1) # Lagged value
        fill_frame['t_l1'] = fill_frame['t'].shift(periods=1) # Lagged value
        missing_periods = (fill_frame['i'] == fill_frame['i_l1']) & (fill_frame['t'] != fill_frame['t_l1'] + 1)
        if np.sum(missing_periods) > 0: # If have data to fill in
            fill_data = []
            for index in fill_frame[missing_periods].index:
                row = fill_frame.iloc[index]
                # Only iterate over missing years
                for t in range(int(row['t_l1']) + 1, int(row['t'])):
                    new_row = {'i': int(row['i']), 'j': fill_j, 'y': fill_y, 't': t}
                    if m: # If m column included
                        new_row['m'] = int(row['m'])
                    fill_data.append(new_row)
            fill_df = pd.concat([pd.DataFrame(fill_row, index=[i]) for i, fill_row in enumerate(fill_data)])
            fill_frame = pd.concat([fill_frame, fill_df]).sort_values(['i', 't']).reset_index(drop=True) # Sort by i, t
        fill_frame.drop(['i_l1', 't_l1'], axis=1, inplace=True)

        return fill_frame
