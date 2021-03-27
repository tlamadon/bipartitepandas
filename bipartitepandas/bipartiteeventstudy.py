'''
Class for a bipartite network in event study form
'''
import pandas as pd
import bipartitepandas as bpd

class BipartiteEventStudy(bpd.BipartiteEventStudyBase):
    '''
    Class for bipartite networks of firms and workers in event study form. Inherits from BipartiteEventStudyBase.

    Arguments:
        *args: arguments for Pandas DataFrame
        col_dict (dict): make data columns readable (requires: wid (worker id), y1 (compensation 1), y2 (compensation 2), f1i (firm id 1), f2i (firm id 2), m (0 if stayer, 1 if mover); optionally include: year_1 (year of observation 1), year_2 (year of observation 2)). Keep None if column names already correct
        **kwargs: keyword arguments for Pandas DataFrame
    '''

    def __init__(self, *args, col_dict=None, **kwargs):
        reference_dict = {'year': ['year_1', 'year_2']}
        # Initialize DataFrame
        super().__init__(*args, reference_dict=reference_dict, col_dict=col_dict, **kwargs)

        # self.logger.info('BipartiteEventStudy object initialized')

    @property
    def _constructor(self):
        '''
        For inheritance from Pandas.
        '''
        return BipartiteEventStudy

    def get_long(self):
        '''
        Return event study data reformatted into long form.

        Returns:
            long_frame (BipartiteLong): BipartiteLong object generated from event study data
        '''
        # Determine whether weight, m, cluster, year columns exist
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
            'year_1': 'year_2',
            'year_2': 'year_1',
            'j1': 'j2',
            'j2': 'j1'
        }

        rename_dict_2 = {
            'f1i': 'fid',
            'y1': 'comp',
            'year_1': 'year',
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
        if years:
            drops += ['year_2']
            astype_dict['year'] = int

        # Append the last row if a mover or stayer over multiple periods (this is because the last observation is only given as an f2i, never as an f1i)
        data_long = pd.DataFrame(self).groupby('wid').apply(lambda a: a.append(a.iloc[-1].rename(rename_dict_1, axis=1)) if a.iloc[0]['m'] == 1 else a) \
            .reset_index(drop=True) \
            .drop(drops, axis=1) \
            .rename(rename_dict_2, axis=1) \
            .astype(astype_dict)

        # Sort columns
        sorted_cols = sorted(data_long.columns, key=bpd.col_order)
        data_long = data_long[sorted_cols]

        long_frame = bpd.BipartiteLong(data_long)
        long_frame.set_attributes(self, no_dict=True)

        return long_frame
