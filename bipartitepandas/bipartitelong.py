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

    def get_es_extended(self, periods_pre=3, periods_post=3):
        '''
        Return Pandas dataframe of event study with periods_pre periods before the transition and periods_post periods after the transition, where transition fulcrums are given by job moves, and the first post-period is given by the job move. Returned dataframe gives worker id, period of transition, income over all periods, and firm cluster over all periods. The function will run .cluster() if no g column exists.

        Arguments:
            periods_pre (int): number of periods before the transition
            periods_post (int): number of periods after the transition

        Returns:
            es_extended_frame (Pandas DataFrame): extended event study generated from long data
        '''
        if not self.col_included('g'):
            # Cluster if no cluster column
            es_extended_frame = pd.DataFrame(self.cluster(), copy=True)
        else:
            es_extended_frame = pd.DataFrame(self, copy=True)

        # Generate how many periods each worker worked
        es_extended_frame['one'] = 1
        es_extended_frame['worker_total_periods'] = es_extended_frame.groupby('i')['one'].transform(sum) # Must faster to use .transform(sum) than to use .transform(len)

        # Keep workers with enough periods (must have at least periods_pre + periods_post periods)
        es_extended_frame = es_extended_frame[es_extended_frame['worker_total_periods'] >= periods_pre + periods_post].reset_index(drop=True)

        # Sort by worker-period
        es_extended_frame.sort_values(['i', 't'])

        # For each worker-period, generate (how many total years - 1) they have worked at that point (e.g. if a worker started in 2005, and had data each year, then 2008 would give 3, 2009 would give 4, etc.)
        es_extended_frame['worker_periods_worked'] = es_extended_frame.groupby('i')['one'].cumsum() - 1
        es_extended_frame.drop('one', axis=1, inplace=True)

        # Find periods where the worker moved firms, which can serve as fulcrums for the event study
        es_extended_frame['moved_firms'] = ((es_extended_frame['i'] == es_extended_frame['i'].shift(periods=1)) & (es_extended_frame['j'] != es_extended_frame['j'].shift(periods=1))).astype(int)

        # Compute valid moves - periods where the worker moved firms, and they also have periods_pre periods before the move, and periods_post periods after (and including) the move
        es_extended_frame['valid_move'] = \
                                es_extended_frame['moved_firms'] & \
                                (es_extended_frame['worker_periods_worked'] >= periods_pre) & \
                                ((es_extended_frame['worker_total_periods'] - es_extended_frame['worker_periods_worked']) >= periods_post)

        # Drop irrelevant columns
        es_extended_frame.drop(['worker_total_periods', 'worker_periods_worked', 'moved_firms'], axis=1, inplace=True)

        # Only keep workers who have a valid move
        es_extended_frame = es_extended_frame[es_extended_frame.groupby('i')['valid_move'].transform(max) > 0]

        # Compute lagged values
        lagged_g = [] # For column order
        lagged_y = [] # For column order
        for i in range(1, periods_pre + 1):
            es_extended_frame['g_l{}'.format(i)] = es_extended_frame['g'].shift(periods=i)
            es_extended_frame['y_l{}'.format(i)] = es_extended_frame['y'].shift(periods=i)
            lagged_g.insert(0, 'g_l{}'.format(i))
            lagged_y.insert(0, 'y_l{}'.format(i))

        # Compute lead values
        lead_g = ['g_f1'] # For column order
        lead_y = ['y_f1'] # For column order
        for i in range(1, periods_post): # No + 1 because y has no shift (i.e. y becomes y_f1)
            es_extended_frame['g_f{}'.format(i + 1)] = es_extended_frame['g'].shift(periods=-i)
            es_extended_frame['y_f{}'.format(i + 1)] = es_extended_frame['y'].shift(periods=-i)
            lead_g.append('g_f{}'.format(i + 1))
            lead_y.append('y_f{}'.format(i + 1))

        # Rename g to g_f1 and y to y_f1
        es_extended_frame.rename({'g': 'g_f1', 'y': 'y_f1'}, axis=1, inplace=True)

        # Keep rows with valid moves
        es_extended_frame = es_extended_frame[es_extended_frame['valid_move'] == 1].reset_index(drop=True)

        # Drop irrelevant columns
        es_extended_frame.drop('valid_move', axis=1, inplace=True)

        # Correct g-column datatypes
        es_extended_frame[lagged_g + lead_g] = es_extended_frame[lagged_g + lead_g].astype(int)

        # Reorder columns
        es_extended_frame = es_extended_frame[['i', 't'] + lagged_g + lead_g + lagged_y + lead_y]

        # Return es_extended_frame
        return es_extended_frame
