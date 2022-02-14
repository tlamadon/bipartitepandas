'''
Class for a bipartite network in collapsed event study form
'''
import bipartitepandas as bpd

class BipartiteEventStudyCollapsed(bpd.BipartiteEventStudyBase):
    '''
    Class for bipartite networks of firms and workers in collapsed event study form (i.e. employment spells are collapsed into a single observation). Inherits from BipartiteEventStudyBase.

    Arguments:
        *args: arguments for BipartiteEventStudyBase
        col_reference_dict (dict): clarify which columns are associated with a general column name, e.g. {'wid': 'wid', 'j': ['j1', 'j2']}
        col_collapse_dict (dict): how to collapse column (None indicates the column should be dropped), e.g. {'y': 'mean'}
        **kwargs: keyword arguments for BipartiteEventStudyBase
    '''

    def __init__(self, *args, col_reference_dict={}, col_collapse_dict={}, **kwargs):
        col_reference_dict = bpd.util.update_dict({'t': ['t11', 't12', 't21', 't22']}, col_reference_dict)
        col_collapse_dict = bpd.util.update_dict({'m': None}, col_collapse_dict)
        # Initialize DataFrame
        super().__init__(*args, col_reference_dict=col_reference_dict, col_collapse_dict=col_collapse_dict, **kwargs)

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

    def uncollapse(self, drop_no_collapse_columns=True, is_sorted=False, copy=True):
        '''
        Return collapsed long data reformatted into long data, by assuming variables constant over spells.

        Arguments:
            drop_no_collapse_columns (bool): if True, columns marked by self.col_collapse_dict as None (i.e. they should be dropped) will not be dropped
            is_sorted (bool): if False, dataframe will be sorted by i (and t, if included). Set to True if already sorted.
            copy (bool): if False, avoid copy

        Returns:
            (BipartiteEventStudy): collapsed event study data reformatted as event study data
        '''
        raise NotImplementedError('.uncollapse() is not implemented for collapsed event study format. Please convert to collapsed long format with the method .to_long().')

    def get_worker_m(self, is_sorted=False):
        '''
        Get NumPy array indicating whether the worker associated with each observation is a mover.

        Arguments:
            is_sorted (bool): used for event study format, does nothing for collapsed event study

        Returns:
            (NumPy Array): indicates whether the worker associated with each observation is a mover
        '''
        return self.loc[:, 'm'].to_numpy() > 0
