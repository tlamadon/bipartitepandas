'''
Class for a bipartite network in collapsed extended event study format.
'''
import bipartitepandas as bpd

class BipartiteExtendedEventStudyCollapsed(bpd.BipartiteExtendedEventStudyBase):
    '''
    Class for bipartite networks of firms and workers in collapsed extended event study form (i.e. employment spells are collapsed into a single observation). Inherits from BipartiteExtendedEventStudyBase.

    Arguments:
        *args: arguments for BipartiteExtendedEventStudyBase
        n_periods (int): number of periods in extended event study
        col_reference_dict (dict or None): clarify which columns are associated with a general column name, e.g. {'i': 'i', 'j': ['j1', 'j2']}; None is equivalent to {}
        col_collapse_dict (dict or None): how to collapse column (None indicates the column should be dropped), e.g. {'y': 'mean'}; None is equivalent to {}
        **kwargs: keyword arguments for BipartiteExtendedEventStudyBase
    '''

    def __init__(self, *args, n_periods=4, col_reference_dict=None, col_collapse_dict=None, **kwargs):
        # Update parameters to be lists/dictionaries instead of None (source: https://stackoverflow.com/a/54781084/17333120)
        if col_reference_dict is None:
            col_reference_dict = {}
        if col_collapse_dict is None:
            col_collapse_dict = {}
        col_reference_dict = bpd.util.update_dict({'t': [f't{t1 + 1}{t2 + 1}' for t1 in range(n_periods) for t2 in range(2)]}, col_reference_dict)
        col_collapse_dict = bpd.util.update_dict({'m': None}, col_collapse_dict)
        # Initialize DataFrame
        super().__init__(*args, n_periods=n_periods, col_reference_dict=col_reference_dict, col_collapse_dict=col_collapse_dict, **kwargs)

        # self.log('BipartiteExtendedEventStudyCollapsed object initialized', level='info')

    @property
    def _constructor(self):
        '''
        For inheritance from Pandas.

        Returns:
            (class): BipartiteExtendedEventStudyCollapsed class
        '''
        return BipartiteExtendedEventStudyCollapsed

    @property
    def _constructor_long(self):
        '''
        For .to_long(), tells BipartiteExtendedEventStudyBase which long format to use.

        Returns:
            (class): BipartiteLongCollapsed class
        '''
        return bpd.BipartiteLongCollapsed

    def uncollapse(self, drop_no_collapse_columns=True, is_sorted=False, copy=True):
        '''
        Return collapsed extended event study data reformatted into event study data, by assuming variables constant over spells.

        Arguments:
            drop_no_collapse_columns (bool): if True, columns marked by self.col_collapse_dict as None (i.e. they should be dropped) will not be dropped
            is_sorted (bool): if False, dataframe will be sorted by i (and t, if included). Returned dataframe will be sorted. Sorting may alter original dataframe if copy is set to False. Set is_sorted to True if dataframe is already sorted.
            copy (bool): if False, avoid copy

        Returns:
            (BipartiteEventStudy): collapsed event study data reformatted as event study data
        '''
        raise NotImplementedError('.uncollapse() is not implemented for collapsed extended event study format. Please convert to collapsed long format with the method .to_long().')

    def get_worker_m(self, is_sorted=False):
        '''
        Get NumPy array indicating whether the worker associated with each observation is a mover.

        Arguments:
            is_sorted (bool): not used for collapsed extended event study format

        Returns:
            (NumPy Array): indicates whether the worker associated with each observation is a mover
        '''
        return self.loc[:, 'm'].to_numpy() > 0
