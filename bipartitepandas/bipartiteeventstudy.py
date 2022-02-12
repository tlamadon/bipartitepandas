'''
Class for a bipartite network in event study form
'''
import bipartitepandas as bpd

class BipartiteEventStudy(bpd.BipartiteEventStudyBase):
    '''
    Class for bipartite networks of firms and workers in event study form. Inherits from BipartiteEventStudyBase.

    Arguments:
        *args: arguments for BipartiteEventStudyBase
        col_reference_dict (dict): clarify which columns are associated with a general column name, e.g. {'wid': 'wid', 'j': ['j1', 'j2']}
        **kwargs: keyword arguments for BipartiteEventStudyBase
    '''

    def __init__(self, *args, col_reference_dict={}, **kwargs):
        col_reference_dict = bpd.util.update_dict({'t': ['t1', 't2']}, col_reference_dict)
        # Initialize DataFrame
        super().__init__(*args, col_reference_dict=col_reference_dict, **kwargs)

        # self.log('BipartiteEventStudy object initialized', level='info')

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

    def get_worker_m(self, is_sorted=False):
        '''
        Get NumPy array indicating whether the worker associated with each observation is a mover.

        Arguments:
            is_sorted (bool): if False, dataframe will be sorted by i in a groupby (but self will not be not sorted). Set to True if already sorted.

        Returns:
            (NumPy Array): indicates whether the worker associated with each observation is a mover
        '''
        return self.groupby('i', sort=(not is_sorted))['m'].transform('max').to_numpy() > 0
