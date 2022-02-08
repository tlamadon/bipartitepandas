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
        **kwargs: keyword arguments for BipartiteEventStudyBase
    '''

    def __init__(self, *args, col_reference_dict={}, **kwargs):
        col_reference_dict = bpd.update_dict({'t': ['t11', 't12', 't21', 't22']}, col_reference_dict)
        # Initialize DataFrame
        super().__init__(*args, col_reference_dict=col_reference_dict, **kwargs)

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

    def get_worker_m(self, is_sorted=False):
        '''
        Get NumPy array indicating whether the worker associated with each observation is a mover.

        Arguments:
            is_sorted (bool): used for event study format, does nothing for collapsed event study

        Returns:
            (NumPy Array): indicates whether the worker associated with each observation is a mover
        '''
        return self.loc[:, 'm'].to_numpy() > 0
