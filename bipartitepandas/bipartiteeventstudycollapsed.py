'''
Class for a bipartite network in collapsed event study form
'''
import numpy as np
import pandas as pd
import bipartitepandas as bpd

class BipartiteEventStudyCollapsed(bpd.BipartiteEventStudyBase):
    '''
    Class for bipartite networks of firms and workers in collapsed event study form (i.e. employment spells are collapsed into a single observation). Inherits from BipartiteEventStudyBase.

    Arguments:
        *args: arguments for Pandas DataFrame
        col_dict (dict): make data columns readable (requires: i (worker id), j1 (firm 1 id), j2 (firm 1 id), y1 (compensation 1), y2 (compensation 2); optionally include: t11 (first period in observation 1 spell), t12 (last period in observation 1 spell), t21 (first period in observation 2 spell), t22 (last period in observation 2 spell), w1 (weight 1), w2 (weight 2), g1 (firm 1 cluster), g2 (firm 2 cluster), m (0 if stayer, 1 if mover)). Keep None if column names already correct
        include_id_reference_dict (bool): if True, create dictionary of Pandas dataframes linking original id values to contiguous id values
        **kwargs: keyword arguments for Pandas DataFrame
    '''

    def __init__(self, *args, col_dict=None, include_id_reference_dict=False, **kwargs):
        reference_dict = {'t': ['t11', 't12', 't21', 't22']}
        # Initialize DataFrame
        super().__init__(*args, reference_dict=reference_dict, col_dict=col_dict, include_id_reference_dict=include_id_reference_dict, **kwargs)

        # self.log('BipartiteEventStudyCollapsed object initialized', level='info')

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

    def _construct_firm_worker_linkages(self, is_sorted=False):
        '''
        Construct numpy array linking firms to worker ids, for use with leave-one-observation-out components.

        Arguments:
            is_sorted (bool): used for event study format, does nothing for collapsed event study

        Returns:
            (tuple of NumPy Array, int): (firm-worker linkages, maximum firm id)
        '''
        move_rows = (self.loc[:, 'm'].to_numpy() > 0)
        base_linkages = self.loc[move_rows, ['i', 'j1']].to_numpy()
        secondary_linkages = self.loc[move_rows, ['i', 'j2']].to_numpy()
        linkages = np.concatenate([base_linkages, secondary_linkages], axis=0)
        max_j = np.max(linkages)

        return linkages, max_j
