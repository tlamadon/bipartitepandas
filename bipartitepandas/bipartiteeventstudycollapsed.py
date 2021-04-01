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
        col_dict (dict): make data columns readable (requires: i (worker id), y1 (compensation 1), y2 (compensation 2), j1 (firm id 1), j2 (firm id 2); optionally include: t11 (first year of observation 1 spell), t12 (last year of observation 1 spell), t21 (first year of observation 2 spell), t22 (last year of observation 2 spell), m (0 if stayer, 1 if mover)). Keep None if column names already correct
        **kwargs: keyword arguments for Pandas DataFrame
    '''

    def __init__(self, *args, col_dict=None, **kwargs):
        columns_opt = ['w']
        reference_dict = {'t': ['t11', 't12', 't21', 't22'], 'w': ['w1', 'w2']}
        col_dtype_dict = {'w': 'float'}
        # Initialize DataFrame
        super().__init__(*args, columns_opt=columns_opt, reference_dict=reference_dict, col_dtype_dict=col_dtype_dict, col_dict=col_dict, **kwargs)

        # self.logger.info('BipartiteEventStudyCollapsed object initialized')

    @property
    def _constructor(self):
        '''
        For inheritance from Pandas.

        Returns:
            (BipartiteEventStudyCollapsed): class
        '''
        return BipartiteEventStudyCollapsed

    @property
    def _constructor_long(self):
        '''
        For get_long(), tells BipartiteEventStudyBase which long format to use.

        Returns:
            (BipartiteLongCollapsed): class
        '''
        return bpd.BipartiteLongCollapsed
