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
        col_dict (dict): make data columns readable (requires: wid (worker id), comp (compensation), fid (firm id), year). Keep None if column names already correct
        **kwargs: keyword arguments for Pandas DataFrame
    '''

    def __init__(self, *args, col_dict=None, **kwargs):
        columns_opt = ['weight']
        reference_dict = {'year': ['year_start', 'year_end'], 'weight': 'weight'}
        col_dtype_dict = {'weight': 'float'}
        # Initialize DataFrame
        super().__init__(*args, columns_opt=columns_opt, reference_dict=reference_dict, col_dtype_dict=col_dtype_dict, col_dict=col_dict, **kwargs)

        # self.logger.info('BipartiteLongCollapsed object initialized')

    @property
    def _constructor(self):
        '''
        For inheritance from Pandas.
        '''
        return BipartiteLongCollapsed

    def get_es(self):
        '''
        Return collapsed long form data reformatted into event study data.

        Returns:
            es_frame (BipartiteEventStudyCollapsed): BipartiteEventStudyCollapsed object generated from collapsed long data
        '''
        # Determine whether m, cluster columns exist
        weighted = self.col_included('weight')
        m = self.col_included('m')
        clustered = self.col_included('j')

        if not m:
            # Generate m column
            self.gen_m()

        # Split workers by movers and stayers
        stayers = pd.DataFrame(self[self['m'] == 0])
        movers = pd.DataFrame(self[self['m'] == 1])
        self.logger.info('workers split by movers and stayers')

        # Add lagged values
        movers = movers.sort_values(['wid', 'year_start'])
        movers['fid_l1'] = movers['fid'].shift(periods=1)
        movers['wid_l1'] = movers['wid'].shift(periods=1) # Used to mark consecutive observations as being for the same worker
        movers['comp_l1'] = movers['comp'].shift(periods=1)
        movers['year_start_l1'] = movers['year_start'].shift(periods=1)
        movers['year_end_l1'] = movers['year_end'].shift(periods=1)
        if weighted:
            movers['weight_l1'] = movers['weight'].shift(periods=1)
        if clustered:
            movers['j_l1'] = movers['j'].shift(periods=1)
        movers = movers[movers['wid'] == movers['wid_l1']]
        movers[['fid_l1', 'year_start_l1', 'year_end_l1']] = movers[['fid_l1', 'year_start_l1', 'year_end_l1']].astype(int) # Shifting adds nans which converts columns into float, but want int

        # Update columns
        stayers = stayers.rename({
            'fid': 'f1i',
            'comp': 'y1',
            'year_start': 'year_start_1',
            'year_end': 'year_end_1',
            'weight': 'w1', # Optional
            'j': 'j1' # Optional
        }, axis=1)
        stayers['f2i'] = stayers['f1i']
        stayers['y2'] = stayers['y1']
        stayers['year_start_2'] = stayers['year_start_1']
        stayers['year_end_2'] = stayers['year_end_1']

        movers = movers.rename({
            'fid_l1': 'f1i',
            'fid': 'f2i',
            'comp_l1': 'y1',
            'comp': 'y2',
            'year_start_l1': 'year_start_1',
            'year_start': 'year_start_2',
            'year_end_l1': 'year_end_1',
            'year_end': 'year_end_2',
            'weight_l1': 'w1', # Optional
            'weight': 'w2', # Optional
            'j_l1': 'j1', # Optional
            'j': 'j2' # Optional
        }, axis=1)

        keep_cols = ['wid', 'y1', 'y2', 'f1i', 'f2i', 'year_start_1', 'year_start_2', 'year_end_1', 'year_end_2', 'm']

        if weighted:
            stayers['w2'] = stayers['w1']
            movers['w1'] = movers['w1'].astype(int)
            keep_cols += ['w1', 'w2']
        if clustered:
            stayers['j2'] = stayers['j1']
            movers['j1'] = movers['j1'].astype(int)
            keep_cols += ['j1', 'j2']

        # Keep only relevant columns
        stayers = stayers[keep_cols]
        movers = movers[keep_cols]
        self.logger.info('columns updated')

        # Merge stayers and movers
        data_es = pd.concat([stayers, movers]).reset_index(drop=True)

        # Sort columns
        sorted_cols = sorted(data_es.columns, key=bpd.col_order)
        data_es = data_es[sorted_cols]

        self.logger.info('data reformatted as event study')

        es_frame = bpd.BipartiteEventStudyCollapsed(data_es)
        es_frame.set_attributes(self, no_dict=True)

        return es_frame
