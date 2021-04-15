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

    def uncollapse(self):
        '''
        Return collapsed long data reformatted into long data, by assuming variables constant over spells.

        Returns:
            long_frame (BipartiteLong): collapsed long data reformatted as BipartiteLong data
        '''
        all_cols = self.included_cols(flat=True)
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
        long_frame.set_attributes(self, no_dict=True)

        return long_frame
