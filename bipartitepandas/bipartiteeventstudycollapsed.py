'''
Class for a bipartite network in collapsed event study form
'''
import pandas as pd
import bipartitepandas as bpd

class BipartiteEventStudyCollapsed(bpd.BipartiteEventStudyBase):
    '''
    Class for bipartite networks of firms and workers in collapsed event study form (i.e. employment spells are collapsed into a single observation). Inherits from BipartiteEventStudyBase.

    Arguments:
        *args: arguments for Pandas DataFrame
        col_dict (dict): make data columns readable (requires: wid (worker id), y1 (compensation 1), y2 (compensation 2), f1i (firm id 1), f2i (firm id 2), m (0 if stayer, 1 if mover); optionally include: year_start_1 (first year of observation 1 spell), year_end_1 (last year of observation 1 spell), year_start_2 (first year of observation 2 spell), year_end_2 (last year of observation 2 spell)). Keep None if column names already correct
        **kwargs: keyword arguments for Pandas DataFrame
    '''

    def __init__(self, *args, col_dict=None, **kwargs):
        columns_opt = ['weight']
        reference_dict = {'year': ['year_start_1', 'year_end_1', 'year_start_2', 'year_end_2'], 'weight': ['w1', 'w2']}
        col_dtype_dict = {'weight': 'float'}
        # Initialize DataFrame
        super().__init__(*args, columns_opt=columns_opt, reference_dict=reference_dict, col_dtype_dict=col_dtype_dict, col_dict=col_dict, **kwargs)

        # self.logger.info('BipartiteEventStudyCollapsed object initialized')

    @property
    def _constructor(self):
        '''
        For inheritance from Pandas.
        '''
        return BipartiteEventStudyCollapsed

    def get_collapsed_long(self):
        '''
        Return collapsed event study data reformatted into collapsed long form.

        Returns:
            collapsedlong_frame (BipartiteCollapsed): BipartiteCollapsed object generated from event study data
        '''
        # Determine whether weight, m, cluster, year columns exist
        weighted = self.col_included('weight')
        m = self.col_included('m')
        clustered = self.col_included('j')
        years = self.col_included('year')

        if not m:
            self.gen_m()

        # Columns to drop
        drops = ['f2i', 'y2']

        rename_dict_1 = {
            'f1i': 'f2i',
            'f2i': 'f1i',
            'y1': 'y2',
            'y2': 'y1',
            'year_start_1': 'year_start_2',
            'year_start_2': 'year_start_1',
            'year_end_1': 'year_end_2',
            'year_end_2': 'year_end_1',
            'w1': 'w2',
            'w2': 'w1',
            'j1': 'j2',
            'j2': 'j1'
        }

        rename_dict_2 = {
            'f1i': 'fid',
            'y1': 'comp',
            'year_start_1': 'year_start',
            'year_end_1': 'year_end',
            'w1': 'weight',
            'j1': 'j'
        }

        astype_dict = {
            'wid': int,
            'fid': int,
            'm': int
        }

        if clustered:
            drops += ['j2']
            astype_dict['j'] = int
        if weighted:
            drops += ['w2']
            astype_dict['weight'] = int
        if years:
            drops += ['year_start_2', 'year_end_2']
            astype_dict['year_start'] = int
            astype_dict['year_end'] = int

        # Append the last row if a mover (this is because the last observation is only given as an f2i, never as an f1i)
        data_collapsed_long = pd.DataFrame(self).groupby('wid').apply(lambda a: a.append(a.iloc[-1].rename(rename_dict_1, axis=1)) if a.iloc[0]['m'] == 1 else a) \
            .reset_index(drop=True) \
            .drop(drops, axis=1) \
            .rename(rename_dict_2, axis=1) \
            .astype(astype_dict)

        # Sort columns
        sorted_cols = sorted(data_collapsed_long.columns, key=bpd.col_order)
        data_collapsed_long = data_collapsed_long[sorted_cols]

        collapsedlong_frame = bpd.BipartiteLongCollapsed(data_collapsed_long)
        collapsedlong_frame.set_attributes(self, no_dict=True)

        return collapsedlong_frame
