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
        col_dict (dict): make data columns readable (requires: i (worker id), j1 (firm 1 id), j2 (firm 1 id), y1 (compensation 1), y2 (compensation 2); optionally include: t1 (time of observation 1), t2 (time of observation 2), g1 (firm 1 cluster), g2 (firm 2 cluster), m (0 if stayer, 1 if mover)). Keep None if column names already correct
        include_id_reference_dict (bool): if True, create dictionary of Pandas dataframes linking original id values to contiguous id values
        **kwargs: keyword arguments for Pandas DataFrame
    '''

    def __init__(self, *args, col_dict=None, include_id_reference_dict=False, **kwargs):
        reference_dict = {'t': ['t1', 't2']}
        # Initialize DataFrame
        super().__init__(*args, reference_dict=reference_dict, col_dict=col_dict, include_id_reference_dict=include_id_reference_dict, **kwargs)

        # self.logger.info('BipartiteEventStudy object initialized')

    @property
    def _constructor(self):
        '''
        For inheritance from Pandas.

        Returns:
            (BipartiteEventStudy): class
        '''
        return BipartiteEventStudy

    @property
    def _constructor_long(self):
        '''
        For get_long(), tells BipartiteEventStudyBase which long format to use.

        Returns:
            (BipartiteLong): class
        '''
        return bpd.BipartiteLong
